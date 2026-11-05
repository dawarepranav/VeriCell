#!/bin/zsh
# Soundtrack alternatives: audio-only renders muxed onto the finished picture (no frames re-rendered).
for variant in "none:no-music:{\"music\":\"none\"}" "ouroboros:ouroboros:{\"music\":\"ouroboros.mp3\",\"musicGain\":1.3}" "dawn:dawn:{\"music\":\"dawn.mp3\",\"musicGain\":3,\"musicFrom\":200}"; do
  key=${variant%%:*}; rest=${variant#*:}; name=${rest%%:*}; props=${rest#*:}
  for pair in "VeriCellDemo:VeriCell-product-demo" "VeriCellShort:VeriCell-30s"; do
    comp=${pair%%:*}; base=${pair#*:}
    npx remotion render $comp out/alt/$base-$name.wav --codec=wav --props="$props" --log=error
    ffmpeg -y -loglevel error -i ../$base.mp4 -i out/alt/$base-$name.wav -map 0:v -map 1:a -c:v copy \
      -af "loudnorm=I=-14:TP=-1.5:LRA=11" -ar 48000 -c:a aac -b:a 192k -movflags +faststart -shortest ../$base-$name.mp4
    echo "$base-$name"; ffmpeg -i ../$base-$name.mp4 -af ebur128=framelog=quiet -vn -f null - 2>&1 | grep -E "^\s+I:"
  done
done
