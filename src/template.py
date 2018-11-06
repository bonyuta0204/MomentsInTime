"""
This module create HTML template for displaying videos
Example
    $python template.py --startI 10 --end 40
create HTML file which presenets images with index between 10 and 40

"""
import argparse
import os

from jinja2 import Environment, FileSystemLoader

from dataloader import MITDataLoader

ROOT_DIR = os.path.join("/", *os.path.abspath(__file__).split("/")[:-2])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=100)
    args = parser.parse_args()
    print(os.path.join(ROOT_DIR, "template"))
    env = Environment(
        loader=FileSystemLoader(
            os.path.join(ROOT_DIR, "template"), encoding='utf8'))
    tpl = env.get_template("template.html")
    train_loader_ = MITDataLoader()
    train_loader = []
    for n in range(args.start, args.end):
        train_loader.append((n, train_loader_[n]))
    print(train_loader)

    html = tpl.render({"train_loader": train_loader})
    print(html)

    with open(os.path.join(ROOT_DIR, "index.html"), "w") as f:
        f.write(html)
