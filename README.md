# ADAM

<a name="readme-top"></a>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#repo-structures">Repository Structures</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

This project is a deep dive into understanding and implementing the ADAM optimizer, a popular optimization algorithm used in training deep learning models. 

The work involved reading and comprehensively understanding the seminal paper "ADAM: A Method for Stochastic Optimization" by Diederik P. Kingma and Jimmy Ba, learning how to create PyTorch-compatible optimizers, and subsequently implementing the ADAM optimizer for use in PyTorch.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple example steps.

### Prerequisites

Install torch. 

### Installation

1. Clone the repo
  ```sh
  git clone https://github.com/jordigb4/Simplex
  ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- USAGE EXAMPLES -->
## Usage

```python
optimizer = torch_ADAM.Adam(model.parameters(),lr=learning_rate)
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Repository Structures

    .
    ├── torch_ADAM.py         # Optimizer class and functions
    ├── pytorch_functions.py  # Optimizing loop functions
    ├── example.ipynb         # Fashion MNIST use case example
    └── README.md

<p align="right">(<a href="#repo-structures">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Jordi Granja Bayot - jordi.granja.i@estudiantat.upc.edu - @jordigb4  

Project Link: [https://github.com/jordigb4/AdamTorch/](https://github.com/jordigb4/AdamTorch/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
