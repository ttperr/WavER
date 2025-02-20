import os
import argparse
import requests
import zipfile

### Config
DATA_FOLDER = 'data' + os.sep

URL_PREFIX = 'https://data.dws.informatik.uni-mannheim.de/benchmarkmatchingtasks/data/'
DATASET_SUFFIX = {
    'abt-buy': 'abt-buy',
    'amazon-google': 'amazon-google',
    'fodors-zagats': 'restaurants_(Fodors-Zagats)',
    'walmart-amazon': 'products_(Walmart-Amazon)',
    'wdc_xlarge_cameras': 'wdc_xlarge_cameras',
    'wdc_xlarge_computers': 'wdc_xlarge_computers',
}
ZIP_FILENAME = 'records.zip'
### End Config


def download_dataset(dataset):
    """
    Downloads the specified dataset from the given URL prefix and saves it in the specified data folder.

    Parameters:
    dataset (str): The name of the dataset to download. It should be one of the keys in the DATASET_SUFFIX dictionary.

    Returns:
    None. The function prints a message indicating the successful download of the dataset.
    """
    dataset_url = DATASET_SUFFIX[dataset]
    if not os.path.exists(DATA_FOLDER + dataset):
        os.makedirs(DATA_FOLDER + dataset)

        for split in ['train', 'val', 'test']:
            url = URL_PREFIX + dataset_url + '/' + 'gs_' + split + '.csv'
            r = requests.get(url, params={'downloadformat': 'csv'})
            with open(DATA_FOLDER + dataset + os.sep + 'gs_' + split + '.csv', 'wb') as f:
                f.write(r.content)

        url = URL_PREFIX + dataset_url + '/' + ZIP_FILENAME
        r = requests.get(url, params={'downloadformat': 'zip'})
        print(url)
        with open(DATA_FOLDER + dataset + os.sep + ZIP_FILENAME, 'wb') as f:
            f.write(r.content)

        print(DATA_FOLDER + dataset + os.sep + ZIP_FILENAME)

        with zipfile.ZipFile(DATA_FOLDER + dataset + os.sep + ZIP_FILENAME, 'r') as z:
            z.extractall(DATA_FOLDER + dataset)

        # Move all file from extracted folder to parent folder
        for file in os.listdir(DATA_FOLDER + dataset + os.sep + 'record_descriptions'):
            os.rename(DATA_FOLDER + dataset + os.sep + 'record_descriptions' + os.sep + file,
                    DATA_FOLDER + dataset + os.sep + file)
        os.rmdir(DATA_FOLDER + dataset + os.sep + 'record_descriptions')
        os.remove(DATA_FOLDER + dataset + os.sep + ZIP_FILENAME)

        print('Downloaded dataset:', dataset, '\n')
        return True
    print('Dataset already exists, skipping download. If you want to re-download, please delete the folder first.')
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='amazon-google', choices=DATASET_SUFFIX.keys(),
                        help='Choose dataset to download')

    hp = parser.parse_args()

    print('Downloading dataset:', hp.dataset, '\n')

    download_dataset(hp.dataset)
