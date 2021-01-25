#! /usr/bin/env sh
cd decomposition
for i in *.wav; do
	ffmpeg -i $i $(echo "../mp3decompose/$(echo "$i" | sed s/\.wav/.mp3/g)")
done
