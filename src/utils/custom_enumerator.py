import datetime
import time
import logging

# Configure Logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Custom enumerator which predicts for loop finish time
def enumerateWithEstimate(iter, desc_str, start_ndx=0):

    iter_len = len(iter)

    log.warning("{} ----/{}, starting".format(
        desc_str,
        iter_len,
    ))

    start_ts = time.time()
    for current_ndx in iter:
        yield current_ndx

        duration_sec = ((time.time() - start_ts)
                        / (current_ndx + 1)
                        * (iter_len)
                        )

        done_dt = datetime.datetime.fromtimestamp(start_ts + duration_sec)

        log.info("{} {:-4}/{}, done at {}".format(
            desc_str,
            current_ndx+1,
            iter_len,
            str(done_dt).rsplit('.', 1)[0],
        ))