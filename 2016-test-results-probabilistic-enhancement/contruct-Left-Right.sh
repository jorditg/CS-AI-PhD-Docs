#!/bin/bash

IFS=$'\n'
echo "Left,Right"
p="1"
for line in $(cat trainLabels.csv); do
  if [ $p == "1" ]; then
     ONE=$(echo $line | cut -d "," -f 2)
     p="2"
  else
     TWO=$(echo $line | cut -d "," -f 2)
     p="1"
     echo $ONE,$TWO
  fi
done

