
selectedFile="$(osascript -l JavaScript -e 'a=Application.currentApplication();a.includeStandardAdditions=true; a.chooseFile({withPrompt:"Choose a file to print:"}).toString()')"
selectedFileBasename=$(basename "${selectedFile}")
printCommand="lpr -P LE302-Colour ~/Downloads/$selectedFileBasename"

echo $selectedFile
function send_file() {
  scp -q $selectedFile afenton@stargate:~/Downloads
}
function print_file() {
  ssh -q afenton@stargate $printCommand
}
function delete_file() {
  ssh -q afenton@stargate "rm ~/Downloads/$selectedFileBasename" 2> /dev/null
}

if [ -z "$selectedFile" ]; then
                echo "No file selected. Printing canceled."
                exit
fi

send_file
echo "File sent to server..."
print_file
echo "Printing started..."
