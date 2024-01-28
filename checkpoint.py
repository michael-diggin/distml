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
    # Optimizer state saved to
    # /directory/epoch/opt.pb
    def __init__(self, directory, frequency, retain=None):
        self.dir = directory
        self.freq = frequency
        self._break = b'<break>' # key word used to mark the end of one string of bytes
        self.num_retain = retain
        self._checkpoints = []

    def should_checkpoint(self, epoch):
        return not epoch%self.freq

    def save_weights(self, epoch, weights, opt_state=None):
        if not os.path.exists(os.path.join(self.dir, str(epoch))):
            os.makedirs(os.path.join(self.dir, str(epoch)))
        proto_weights = serialize.weights_to_proto(weights)
        bin_data = [pw.SerializeToString() for pw in proto_weights]
        bd = self._break.join(bin_data)
        file_name = os.path.join(self.dir, str(epoch), "weights.pb")
        with open(file_name, 'wb') as f:
            f.write(bd)

        # for the opt state, create weights list for input
        if opt_state:
            opt_weights = []
            for i in range(len(opt_state.keys())):
                opt_weights.append(opt_state[str(i)])
            opt_proto_weights = serialize.opt_weights_to_proto(opt_weights)
            bin_data = [pw.SerializeToString() for pw in opt_proto_weights]
            bd = self._break.join(bin_data)
            opt_file = os.path.join(self.dir, str(epoch), "opt.pb")
            with open(opt_file, 'wb') as f:
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
            return 0, None, None
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

        opt_file = os.path.join(self.dir, latest, "opt.pb")
        if not os.path.exists(opt_file):
            return epoch, weights_arrays, None

        with open(opt_file, 'rb') as f:
            opt_data = f.read()
        opt_weights = opt_data.split(self._break)
        opt_proto = [train_pb2.Ndarray() for _ in range(len(opt_weights))]
        for pw, b in zip(opt_proto, opt_weights):
            pw.ParseFromString(b)
        opt_state = serialize.opt_weights_from_proto(opt_proto)
        return epoch, weights_arrays, opt_state

    def _delete_oldest_checkpoint(self):
        oldest = self._checkpoints.pop(0)
        fname = os.path.join(self.dir, str(oldest), "weights.pb")
        opt_name = os.path.join(self.dir, str(oldest), "opt.pb")
        os.remove(fname)
        os.remove(opt_name)
        os.rmdir(os.path.join(self.dir, str(oldest)))
