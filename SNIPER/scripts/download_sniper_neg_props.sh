#-----------------------------------------------------
# SNIPER: Efficient Multi-Scale Training
# Downloading pre-computed props for negative sampling
#-----------------------------------------------------
#!/bin/bash
file_name="proposals.tar.gz"
cur_dir=${PWD##*/}
target_dir="./data/"

if [ $cur_dir = "scripts" ]; then
   target_dir="../data/"
fi

if [ ! -f "${target_dir}${file_name}" ]; then
   echo "Downloading ${file_name}..."
   curl -LA "github `date`" https://bit.ly/3j8SOl7 --output "${target_dir}${file_name}"
   echo "Done!"
else
   echo "File already exists!"
fi

echo "Unzipping the file..."
tar -xvzf "${target_dir}${file_name}" -C ${target_dir}

echo "Cleaning up..."
rm "${target_dir}${file_name}"
echo "All done!"
