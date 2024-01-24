import serialize
import os
import train_pb2


class NoopCheckpoint():
    def __init__(self):
        pass

    def load_latest_weights(self):
        return 0, None
    
    def should_checkpoint(self, epoch):
        return False
    
    def save_weights(self, epoch, weights):
        return

class FileCheckpoint():
    # Weights get saved to
    # /directory/epoch/weights.pb
    def __init__(self, directory, frequency, retain=None):
        self.dir = directory
        self.freq = frequency
        self._break = b'<break>' # key word used to mark the end of one string of bytes
        self.num_retain = retain
        self._checkpoints = []

    def should_checkpoint(self, epoch):
        return not epoch%self.freq

    def save_weights(self, epoch, weights):
        if not os.path.exists(os.path.join(self.dir, str(epoch))):
            os.makedirs(os.path.join(self.dir, str(epoch)))
        proto_weights = serialize.weights_to_proto(weights)
        bin_data = [pw.SerializeToString() for pw in proto_weights]
        bd = self._break.join(bin_data)
        file_name = os.path.join(self.dir, str(epoch), "weights.pb")
        with open(file_name, 'wb') as f:
            f.write(bd)
        
        if self.num_retain:
            self._checkpoints.append(epoch)
            if len(self._checkpoints) > self.num_retain:
                self._delete_oldest_checkpoint()

    def load_latest_weights(self):
        if not os.path.exists(self.dir):
            return 0, None
        files = os.listdir(self.dir)
        if len(files) == 0:
            return 0, None
        files.sort(key=lambda x: int(x))
        latest = files[-1]
        epoch = int(latest)
        with open(os.path.join(self.dir, latest, "weights.pb"), 'rb') as f:
            bin_data = f.read()
        bin_weights = bin_data.split(self._break)
        proto_weights = [train_pb2.Ndarray() for _ in range(len(bin_weights))]
        for pw, b in zip(proto_weights, bin_weights):
            pw.ParseFromString(b)
        weights_arrays = serialize.weights_from_proto(proto_weights)
        return epoch, weights_arrays

    def _delete_oldest_checkpoint(self):
        oldest = self._checkpoints.pop(0)
        fname = os.path.join(self.dir, str(oldest), "weights.pb")
        os.remove(fname)
        os.rmdir(os.path.join(self.dir, str(oldest)))
