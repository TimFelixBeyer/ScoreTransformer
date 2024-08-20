# Score Transformer (fork)

This is an upgraded implementation of the score similarity metric introduced in http://www.dollos.it/assets/pdf/ismir2017.pdf and further extended in https://arxiv.org/abs/2112.00355.

Changes:
- Much faster alignment procedure using `numba`
- Added support for expressions and ornaments
- Only compute metrics for notes (not rests)
- Unroll repeats
- Strip all ties

Please note that this version requires a modified version of `music21`, which can be found here: https://github.com/TimFelixBeyer/music21

## Usage

```python
from score_transformer import score_similarity
print(score_similarity(path_a, path_b)
```


## Installation

```bash
pip install .
```

## Usage

```python
from score_transformer import score_similarity
score_similarity("test1.musicxml", "test2.musicxml")
# or
import music21
score_similarity(music21.converter.parse("test1.musicxml"), music21.converter.parse("test2.musicxml"))
```


## Old README:
This is the official repository for "Score Transformer" (ACM Multimedia Asia 2021 / ISMIR2021 LBD).

[Paper](https://arxiv.org/abs/2112.00355) | [Short paper](https://archives.ismir.net/ismir2021/latebreaking/000032.pdf) | [Project page](https://score-transformer.github.io/)

<!--
- [Score Transformer: Generating Musical Scores from Note-level Representation](https://arxiv.org/abs/2112.00355) (ACM Multimedia Asia 2021)
- [Score Transformer: Transcribing Quantized MIDI into Comprehensive Musical Score](https://archives.ismir.net/ismir2021/latebreaking/000032.pdf) (ISMIR2021 LBD)

Project page: https://score-transformer.github.io/
-->

## Overview

This repository provides:
- [**Tokenization tools**](tokenization_tools) between MusicXML scores and score tokens
- A [**metric**](metric) used in the papers

## Citation
If you find this repository helpful, please consider citing our paper:
```
@inproceedings{suzuki2021,
 author = {Suzuki, Masahiro},
 title = {Score Transformer: Generating Musical Score from Note-level Representation},
 booktitle = {Proceedings of the 3rd ACM International Conference on Multimedia in Asia},
 year = {2021},
 pages = {31:1--31:7},
 doi = {10.1145/3469877.3490612}
}
```
