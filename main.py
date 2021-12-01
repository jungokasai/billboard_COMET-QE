import argparse, json
from comet import download_model, load_from_checkpoint

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--src', type=str, metavar='N',
                    help='source file')
parser.add_argument('--hyp', type=str, metavar='N',
                    help='hypothesis file')
parser.add_argument('--refs', type=str, metavar='N',
                    help='reference file')
parser.add_argument('--outfile', type=str, metavar='N',
                    help='output file')
args = parser.parse_args()

def read_jsonl(infile, extract_key=None):
    f = open(infile, 'r')
    if extract_key is None:
        out = [json.loads(line.strip()) for line in f]
    else:
        out = [json.loads(line.strip())[extract_key] for line in f]
    f.close()
    return out

def create_data(src, hyp, refs):
    assert len(src) == len(hyp)
    assert len(hyp) == len(refs)
    # comet-qe just uses one reference per instance
    out = [{'src': src[i], 'mt': hyp[i], 'ref': refs[i][0]} for i in range(len(src))]
    return out
        

def score(src, hyp, refs, outfile):
    src = read_jsonl(src, 'src')
    hyp = read_jsonl(hyp, 'hyp')
    refs = read_jsonl(refs, 'refs')
    #model_path = download_model("wmt20-comet-da")
    model_path = download_model("wmt20-comet-qe-da")
    #model_path = download_model("emnlp20-comet-rank")
    model = load_from_checkpoint(model_path)
    data = create_data(src, hyp, refs)
    scores, sys_score = model.predict(data)
    with open(outfile, 'wt') as fout:
        for score in scores:
            fout.write(str(score))
            fout.write('\n')


if __name__ == '__main__':
    import torch
    print(torch.cuda.is_available())
    score(args.src, args.hyp, args.refs, args.outfile)
