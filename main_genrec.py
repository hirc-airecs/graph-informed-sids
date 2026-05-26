import argparse

from genrec.utils import parse_command_line_args, get_pipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='TIGER', help='Model name')
    parser.add_argument('--dataset', type=str, default='AmazonReviews2014', help='Dataset name')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
    parser.add_argument(
        '--resume-training', action='store_true', help='If checkpoint is provided, whether to resume training or evaluate it'
    )
    parser.add_argument('--config-file', type=str, default=None, help='Config file to override model/dataset configs')
    parser.add_argument('--test-nrows', type=int, default=None, help='Whether to load subset of dataset users')
    return parser.parse_known_args()


if __name__ == '__main__':
    args, unparsed_args = parse_args()
    command_line_configs = parse_command_line_args(unparsed_args)

    pipeline = get_pipeline(args.model)(
        model_name=args.model,
        dataset_name=args.dataset,
        checkpoint_path=args.checkpoint,
        config_file=args.config_file,
        config_dict=command_line_configs,
        test_nrows=args.test_nrows,
        resume_training=args.resume_training
    )
    pipeline.run()
