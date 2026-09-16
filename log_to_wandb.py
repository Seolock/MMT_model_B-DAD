import re
import wandb

def parse_fairseq_log(log_path):

    results = {}

    train_pat = re.compile(
        r"train.*epoch\s+(\d+).*loss=([\d\.]+)"
    )
    valid_pat = re.compile(
        r"valid.*epoch\s+(\d+).*loss\s+([\d\.]+).*?\| bleu\s+([\d\.]+)\s+\|"
    )

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            # ---------- TRAIN ----------
            m = train_pat.search(line)
            if m:
                epoch = int(m.group(1))
                train_loss = float(m.group(2))

                if epoch not in results:
                    results[epoch] = {}

                results[epoch]["epoch"] = epoch
                results[epoch]["train_loss"] = train_loss

            # ---------- VALID ----------
            m = valid_pat.search(line)
            if m:
                epoch = int(m.group(1))
                valid_loss = float(m.group(2))
                bleu = float(m.group(3))

                if epoch not in results:
                    results[epoch] = {}

                results[epoch]["epoch"] = epoch
                results[epoch]["valid_loss"] = valid_loss
                results[epoch]["bleu"] = bleu

    final = []
    for ep in sorted(results.keys()):
        final.append(results[ep])

    return final


def upload_to_wandb(results, project_name, line_name):
    wandb.init(
        entity="22100356-handong-global-university",
        project=project_name,
        name=line_name
    )

    for row in results:
        wandb.log(row)

    wandb.finish()


if __name__ == "__main__":
    log_file = "checkpoints/multi30k-en2de-vit_base_patch16_384-decoder_test_auged_5/train.log"
    parsed = parse_fairseq_log(log_file)
    print("Parsed Results:")
    for r in parsed:
        print(r)

    upload_to_wandb(parsed,"B_test_auged","uf4_wd0.01")
