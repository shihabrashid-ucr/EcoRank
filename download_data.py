# Credits: The design of the source code follows from the DPR download_data.py script
# Credits: This source code is taken from UPR code repository: https://github.com/DevSinghSachan/unsupervised-passage-reranking


"""
 Command line tool to download various preprocessed data sources for UPR.
"""
import argparse
import tarfile
import os
import pathlib
from subprocess import Popen, PIPE


RESOURCES_MAP = {
    # Wikipedia
    "data.wikipedia-split.psgs_w100": {
        "dropbox_url": "https://www.dropbox.com/s/bezryc9win2bha1/psgs_w100.tar.gz",
        "original_ext": ".tsv",
        "compressed": True,
        "desc": "Entire wikipedia passages set obtain by splitting all pages into 100-word segments (no overlap)",
    },

    # BM25
    "data.retriever-outputs.bm25.webq-test": {
        "dropbox_url": "https://www.dropbox.com/s/yp3zp0brotckzlz/webq-test.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from BM25 retriever for WebQuestions test set.",
    },
    "data.retriever-outputs.bm25.nq-test": {
        "dropbox_url": "https://www.dropbox.com/s/ml2lnt34ktjgft6/nq-test.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from BM25 retriever for Natural Questions Open test set.",
    }
    
}


def unpack(tar_file: str, out_path: str):
    print("Uncompressing %s", tar_file)
    input = tarfile.open(tar_file, "r:gz")
    input.extractall(out_path)
    input.close()
    print(" Saved to %s", out_path)


def download_resource(
    dropbox_url: str, original_ext: str, compressed: bool, resource_key: str, out_dir: str
) -> None:
    print("Requested resource from %s", dropbox_url)
    path_names = resource_key.split(".")

    if out_dir:
        root_dir = out_dir
    else:
        # since hydra overrides the location for the 'current dir' for every run and we don't want to duplicate
        # resources multiple times, remove the current folder's volatile part
        root_dir = os.path.abspath("./")
        if "/outputs/" in root_dir:
            root_dir = root_dir[: root_dir.index("/outputs/")]

    print("Download root_dir %s", root_dir)

    save_root = os.path.join(root_dir, "downloads", *path_names[:-1])  # last segment is for file name

    pathlib.Path(save_root).mkdir(parents=True, exist_ok=True)

    local_file_uncompressed = os.path.abspath(os.path.join(save_root, path_names[-1] + original_ext))
    print("File to be downloaded as %s", local_file_uncompressed)

    if os.path.exists(local_file_uncompressed):
        print("File already exist %s", local_file_uncompressed)
        return

    local_file = os.path.abspath(os.path.join(save_root, path_names[-1] + (".tar.gz" if compressed else original_ext)))

    process = Popen(['wget', dropbox_url, '-O', local_file], stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    print(stdout.decode("utf-8"))
    # print(stderr.decode("utf-8"))
    print("Downloaded to %s", local_file)

    if compressed:
        # uncompressed_path = os.path.join(save_root, path_names[-1])
        unpack(local_file, save_root)
        os.remove(local_file)
    return



def download(resource_key: str, out_dir: str = None):
    if resource_key not in RESOURCES_MAP:
        # match by prefix
        resources = [k for k in RESOURCES_MAP.keys() if k.startswith(resource_key)]
        print("matched by prefix resources: %s", resources)
        if resources:
            for key in resources:
                download(key, out_dir)
        else:
            print("no resources found for specified key")
        return []
    download_info = RESOURCES_MAP[resource_key]

    dropbox_url = download_info["dropbox_url"]

    if isinstance(dropbox_url, list):
        for i, url in enumerate(dropbox_url):
            download_resource(
                url,
                download_info["original_ext"],
                download_info["compressed"],
                "{}_{}".format(resource_key, i),
                out_dir,
            )
    else:
        download_resource(
            dropbox_url,
            download_info["original_ext"],
            download_info["compressed"],
            resource_key,
            out_dir,
        )
    return


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        default="./",
        type=str,
        help="The output directory to download file",
    )
    parser.add_argument(
        "--resource",
        type=str,
        help="Resource name. See RESOURCES_MAP for all possible values",
    )
    args = parser.parse_args()
    if args.resource:
        download(args.resource, args.output_dir)
    else:
        print("Please specify resource value. Possible options are:")
        for k, v in RESOURCES_MAP.items():
            print("Resource key=%s  :  %s", k, v["desc"])


if __name__ == "__main__":
    main()