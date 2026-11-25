"""Training/eval entry for MaskFormer on PanNuke. Registers the converted COCO
datasets, then delegates to train_net.main. New file; does not modify train_net."""
import train_net
from detectron2.engine import launch, default_argument_parser


def main(args):
    import register_pannuke  # noqa: F401  (register inside each spawned process)
    return train_net.main(args)


if __name__ == "__main__":
    import register_pannuke  # noqa: F401  (register in the main process)
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(main, args.num_gpus, num_machines=args.num_machines,
           machine_rank=args.machine_rank, dist_url=args.dist_url, args=(args,))
