"""
This module create HTML template for displaying videos
Example
    $python template.py --startI 10 --end 40
create HTML file which presenets images with index between 10 and 40

"""
import os

from jinja2 import Environment, FileSystemLoader

from dataloader import MITDataLoader

ROOT_DIR = os.path.join("/", *os.path.abspath(__file__).split("/")[:-2])


class HtmlBuilder():
    def __init__(self):
        self.env = Environment(
            loader=FileSystemLoader(
                os.path.join(ROOT_DIR, "template"), encoding='utf8'))
        self.tpl = self.env.get_template("template.html")

    def __call__(self, data_iter, save=True, save_path="index.html"):
        html = self.tpl.render({"data_iter": data_iter})
        print(html)
        if save:
            with open(os.path.join(ROOT_DIR, save_path), "w") as f:
                f.write(html)


if __name__ == "__main__":
    builder = HtmlBuilder()
    dataloader = MITDataLoader()
    index_filtered = dataloader.filter_data(
        lambda data: data.has_object_label("cat"))
    builder(index_filtered)
