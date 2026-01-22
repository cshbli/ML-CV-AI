## Split one huge folder to multiple sub folders
- each sub folder has 28000 image files

```bash
i=0; 
find . -maxdepth 1 -type f | while read -r file; do 
    if [ $((i % 28000)) -eq 0 ]; then 
        dir="batch_$((i / 28000))"; 
        mkdir -p "$dir"; 
    fi; 
    mv "$file" "$dir/"; 
    i=$((i + 1)); 
done
```

## Create one tar file for each sub folder

```bash
for dir in batch_2*/; do
    # Remove the trailing slash from the directory name for the filename
    dirname=${dir%/}
    
    echo "Processing $dirname..."
    
    # Create the tar file
    tar -cvf "${dirname}.tar" "$dirname"
    
    echo "Finished $dirname.tar"
done
```

## Batch extraction 

- Run this command from the directory where the .tar files are located:
```bash
for file in batch_*.tar; do
    echo "Extracting $file..."
    tar --warning=no-unknown-keyword -xf "$file"
    echo "Done with $file"
done
```

By default, the command above will extract the folders into the current directory.

If you want to extract to the External Drive: specify the destination using the -C (change directory) flag:
```bash
# Example: Extracting to a specific folder on your USB drive
for file in batch_2*.tar; do
    echo "Extracting $file to External Drive..."
    tar --warning=no-unknown-keyword -xf "$file" -C /Volumes/data/destination_folder/
done
```

## Ubuntu open tar file which is created on Mac 
```
tar --warning=no-unknown-keyword -xf your_file.tar
```

## MacOS force eject/unmount USB drive
```
diskutil unmount force /Volumes/YourDriveName
```