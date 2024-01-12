set abcfile=%1
set suffix=%abcfile:~0,-4%
echo "%suffix%.mid" 
abc2midi %abcfile% -o "%suffix%.mid"
timidity "%suffix%.mid" -Ow "%suffix%.wav"
