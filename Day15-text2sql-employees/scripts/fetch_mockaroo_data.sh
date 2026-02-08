#!/bin/bash

mkdir -p data

curl -L "https://api.mockaroo.com/api/dde01370?count=1000&key=11149690" \
  -o data/employees.csv

curl -L "https://api.mockaroo.com/api/8ba6f630?count=100&key=11149690" \
  -o data/departments.csv
