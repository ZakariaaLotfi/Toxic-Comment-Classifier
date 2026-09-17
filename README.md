# RoBERTa Toxic Comment Classification with Reply-Based Features

## Overview

This project was part of an NSF-funded Research Experience for Undergraduates (REU), and investigates if using the replies to comments can provide clues to the original comment's intent and thus help improve toxic comment detection.

## Approach

- **Model**: RoBERTa-based classifier with reply features
- **Dataset**: CAD (Conversational Abuse Dataset)

## Repository Structure

```
/
├── data/          # CSVs and Python scripts for data processing and analysis
├── roberta/       # Model training and evaluation
└── README.md
```

## References

Key papers for this work:

 - B. Vidgen, D. Nguyen, H. Margetts, P. Rossini, and R. Tromble, Introducing CAD: the Contextual Abuse Dataset, in Proc. 2021 Conf. North American Chapter Assoc. Comput. Linguistics: Human Language Technologies (NAACL-HLT), 2021, pp. 2289--2303.
 - S. Berezin, R. Farahbakhsh, and N. Crespi, From intrinsic toxicity to reception-based toxicity: A contextual framework for prediction and evaluation, arXiv:2503.16072, 2025.
 - Himanshu, M. Kumar, V. Choudhary, and Y. Nishal, Using discussion thread context in sentiment analysis for improving cyberbullying detection, in Proc. 2023 3rd Asian Conf. Innovation Technol. (ASIANCON), Pune, India, 2023, pp. 1--4.
 - Justine Zhang, Jonathan Chang, Cristian Danescu-Niculescu-Mizil, Lucas Dixon, Yiqing Hua, Dario Taraborelli, and Nithum Thain. 2018. Conversations Gone Awry: Detecting Early Signs of Conversational Failure. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1350–1361, Melbourne, Australia. Association for Computational Linguistics.

## Acknowledgments

This work was supported by the National Science Foundation under Award #2447577. Thank you to my mentors for their guidance throughout this project.

## License

MIT
