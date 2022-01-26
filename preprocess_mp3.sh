usage="$(basename "$0") SOURCE_DIR DEST_DIR [-h] -- Convert mp3 to wav, to be fed into the model

where:
    -h  show this help text"

while getopts ':hs:' option; do
  case "$option" in
    h) echo "$usage"
       exit
       ;;
  esac
done
shift $((OPTIND - 1))

if [ -z "$1" ] || [ -z "$2" ] || [ ! -d "$1" ] || [ ! -d "$2" ]; then
	echo -e "ERROR: Source or Destination directory invalid\n"
    echo "$usage" >&2
	exit 1
fi

cd $2
rm *
cd -
find $1 -name "*.mp3" | xargs -I here gcp --backup=t here $2
cd $2
ls -v | cat -n | while read n f; do mv "$f" "$n.mp34"; done
ls -v | cat -n | while read n f; do mv "$f" "$n.mp3"; done
for file in *.mp3; do sox "${file}" -r 16000 -c 1 "${file/.mp3/_1.wav}" trim 0 3; done
for file in *.wav; do soxi -D "${file}" | xargs -I len echo '3-' len | bc | xargs -I len2 sox "${file}" "${file/_1.wav/.wav}" pad 0 len2; done
rm *.mp3
rm *_1.wav
