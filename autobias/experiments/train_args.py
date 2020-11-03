MODES = ["none", "mce", "oracle", "adv", "noci", "nobp"]


def add_train_args(parser, default_entropy_penalty, default_adv_penalty,
                   default_batch_size, default_epochs, lc_weight_default=0.2, entropy_w=True):
  parser.add_argument("--output_dir", help="Directory to save the model")
  parser.add_argument("--mode", choices=MODES, default="mce",
                      help="Ablation mode, default to MCE")
  parser.add_argument("--lc_weight", type=float, default=lc_weight_default,
                      help="Weight on lower capacity model's loss, ignored for oracle/none mode")
  parser.add_argument("--adversary_loss", type=float, default=default_adv_penalty,
                      help="Weight on the negative adversary loss for adv mode")
  if entropy_w:
    parser.add_argument("--entropy_penalty", type=float, default=default_entropy_penalty,
                        help="Entropy penalty to use with oracle mode")

  parser.add_argument("--debug", action="store_true",
                      help="Run with reduced versions of the target model/datasets")
  parser.add_argument("--seed", type=int, help="Random seed to use")
  parser.add_argument("--init_only", action="store_true",
                      help="Save the configuration without training the model")
  parser.add_argument("--fp16", action="store_true", help="Run in fp16 mode")
  parser.add_argument("--nocuda", action="store_true", help="Run without CUDA")
  parser.add_argument("--n_processes", type=int, default=4,
                      help="N processes to use when tokenizing")
  parser.add_argument("--batch_size", type=int, default=default_batch_size)
  parser.add_argument("--epochs", type=int, default=default_epochs)

