language: julia

os:
  - linux
julia:
  - 1
  - nightly
matrix:
  allow_failures:
    - julia: nightly
notifications:
  email: false