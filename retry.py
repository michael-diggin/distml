import time
import grpc
import logging

# retry decorator
# if max_retries = 0 will retry indefinitely
# will retry on the provided gRPC status codes
def retry_on_statuscode(max_retries, wait_time, status_codes):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while max_retries == 0 or retries < max_retries:
                try:
                    result = func(*args, **kwargs)
                    return result
                except grpc.RpcError as grpc_error:
                    if status_codes and grpc_error.code() in status_codes:
                        retries += 1
                        logging.info(f"Retrying {func}...")
                        time.sleep(wait_time)
                    else:
                        raise grpc_error
                except Exception as e:
                    logging.error(f"Got an error: {e}")
                    raise e
            else:
              raise Exception(f"Max retries of function {func} exceeded")
        return wrapper
    return decorator