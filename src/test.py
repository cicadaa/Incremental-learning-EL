import time
import logging
import threading


def thread_function(name):
    logging.info("Thread %s: starting", name)
    time.sleep(10)
    logging.info("Thread %s: finishing", name)


if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    logging.info("Main    : before creating thread")
    # TODO: rename varibale to make it understandable
    x = threading.Thread(target=thread_function, args=(1,))
    logging.info("Main    : before running thread")
    x.start()
    logging.info("Main    : wait for the thread to finish")
    i = 0
    while i < 12:
        time.sleep(1)
        # TODO: use logging as other place
        print(i)
        i += 1
    # x.join()
    logging.info("Main    : all done")
