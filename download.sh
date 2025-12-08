#!/bin/bash

wget "https://github.com/asabenhur/CS545/raw/refs/heads/master/d2l.py"

archive="yws3v3mwx3-4.zip"
wget "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/$archive"
unzip $archive

dataset="EarVN1.0"
unrar x "$dataset/$dataset dataset.rar"

rm $archive
rm -r $dataset

mv "$dataset dataset" "$dataset"
