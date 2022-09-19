# -*- coding: utf-8 -*-

from paddle_serving_server.web_service import WebService, Op


def reload_logger(log_dir):
    import logging.config
    import os

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger_config = {
        "version": 1,
        "formatters": {
            "normal_fmt": {
                "format":
                    "%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s",
            },
            "tracer_fmt": {
                "format": "%(asctime)s %(message)s",
            },
        },
        "handlers": {
            "f_pipeline.log": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "WARNING",
                "formatter": "normal_fmt",
                "filename": os.path.join(log_dir, "pipeline.log"),
                "maxBytes": 10000,
                "backupCount": 5,
            },
            "f_pipeline.log.wf": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "WARNING",
                "formatter": "normal_fmt",
                "filename": os.path.join(log_dir, "pipeline.log.wf"),
                "maxBytes": 10000,
                "backupCount": 10,
            },
            "f_tracer.log": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "WARNING",
                "formatter": "tracer_fmt",
                "filename": os.path.join(log_dir, "pipeline.tracer"),
                "maxBytes": 10000,
                "backupCount": 5,
            },
        },
        "loggers": {
            # propagate = True
            ".".join(__name__.split(".")[:-1] + ["profiler"]): {
                "level": "INFO",
                "handlers": ["f_tracer.log"],
            },
        },
        "root": {
            "level": "DEBUG",
            "handlers": ["f_pipeline.log", "f_pipeline.log.wf"],
        },
    }
    logging.config.dictConfig(logger_config)


def convert_example(example,
                    tokenizer,
                    max_seq_length=512,
                    pad_to_max_seq_len=False):
    result = []
    for text in example:
        encoded_inputs = tokenizer(
            text=text,
            max_seq_len=max_seq_length,
            pad_to_max_seq_len=pad_to_max_seq_len)
        input_ids = encoded_inputs["input_ids"]
        token_type_ids = encoded_inputs["token_type_ids"]
        attention_mask = [1] * len(input_ids)
        result += [input_ids, token_type_ids, attention_mask]
    return result


class BertOp(Op):
    tokenizer = None

    def init_op(self):
        from paddlenlp.transformers.bert.faster_tokenizer import BertFasterTokenizer
        self.tokenizer = BertFasterTokenizer.from_pretrained('bert-base-chinese')

    def set_dynamic_shape_info(self):
        min_input_shape = {
            "input_ids": [1, 1],
            "token_type_ids": [1, 1],
            "attention_mask": [1, 1],
            "tmp_2": [1, 1]
        }
        max_input_shape = {
            "input_ids": [10, 512],
            "token_type_ids": [10, 512],
            "attention_mask": [10, 512],
            "tmp_2": [10, 512]
        }
        opt_input_shape = {
            "input_ids": [1, 512],
            "token_type_ids": [1, 512],
            "attention_mask": [1, 512],
            "tmp_2": [1, 512]
        }

        self.dynamic_shape_info = {
            "min_input_shape": min_input_shape,
            "max_input_shape": max_input_shape,
            "opt_input_shape": opt_input_shape,
        }

    def preprocess(self, input_dicts, data_id=0, log_id=0):
        from paddlenlp.data import Tuple, Pad
        (_, input_dict), = input_dicts.items()
        batch_size = len(input_dict.keys())
        examples = []
        for i in range(batch_size):
            input_ids, segment_ids, attention_mask = convert_example([input_dict[str(i)]],
                                                                     self.tokenizer)
            examples.append((input_ids, segment_ids, attention_mask))
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=self.tokenizer.pad_token_id),  # input
            Pad(axis=0, pad_val=self.tokenizer.pad_token_id),  # segment
            Pad(axis=0, pad_val=self.tokenizer.pad_token_id),  # attention mask
        ): fn(samples)
        input_ids, segment_ids, attention_mask = batchify_fn(examples)

        feed_dict = {'input_ids': input_ids,
                     'token_type_ids': segment_ids,
                     'attention_mask': attention_mask}
        return feed_dict, False, None, ""

    def postprocess(self, input_data, fetch_data, data_id=0, log_id=0):
        import scipy.special
        for key in fetch_data.keys():
            fetch_data[key] = scipy.special.softmax(fetch_data[key], axis=-1)
        return fetch_data, None, ""


class CodeReviewService(WebService):
    def get_pipeline_response(self, read_op):
        bert_op = BertOp(name="bert", input_ops=[read_op])
        return bert_op


reload_logger("logs")
ocr_service = CodeReviewService(name="bert")
ocr_service.prepare_pipeline_config("config.yml")
ocr_service.run_service()
