{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "411e901c-2a20-4543-a110-56032c9a698a",
   "metadata": {},
   "source": [
    "# Finetune bert classifier for sentiment classification\n",
    "Example from https://huggingface.co/docs/transformers/training"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "c9dc6ace-e8af-4ab1-8a0d-cc15bb6572f7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: transformers[torch] in /opt/conda/lib/python3.11/site-packages (4.36.2)\n",
      "Requirement already satisfied: filelock in /opt/conda/lib/python3.11/site-packages (from transformers[torch]) (3.9.0)\n",
      "Requirement already satisfied: huggingface-hub<1.0,>=0.19.3 in /opt/conda/lib/python3.11/site-packages (from transformers[torch]) (0.20.1)\n",
      "Requirement already satisfied: numpy>=1.17 in /opt/conda/lib/python3.11/site-packages (from transformers[torch]) (1.26.0)\n",
      "Requirement already satisfied: packaging>=20.0 in /opt/conda/lib/python3.11/site-packages (from transformers[torch]) (23.2)\n",
      "Requirement already satisfied: pyyaml>=5.1 in /opt/conda/lib/python3.11/site-packages (from transformers[torch]) (6.0.1)\n",
      "Requirement already satisfied: regex!=2019.12.17 in /opt/conda/lib/python3.11/site-packages (from transformers[torch]) (2023.12.25)\n",
      "Requirement already satisfied: requests in /opt/conda/lib/python3.11/site-packages (from transformers[torch]) (2.31.0)\n",
      "Requirement already satisfied: tokenizers<0.19,>=0.14 in /opt/conda/lib/python3.11/site-packages (from transformers[torch]) (0.15.0)\n",
      "Requirement already satisfied: safetensors>=0.3.1 in /opt/conda/lib/python3.11/site-packages (from transformers[torch]) (0.4.1)\n",
      "Requirement already satisfied: tqdm>=4.27 in /opt/conda/lib/python3.11/site-packages (from transformers[torch]) (4.66.1)\n",
      "Requirement already satisfied: torch!=1.12.0,>=1.10 in /opt/conda/lib/python3.11/site-packages (from transformers[torch]) (2.1.1+cu118)\n",
      "Requirement already satisfied: accelerate>=0.21.0 in /opt/conda/lib/python3.11/site-packages (from transformers[torch]) (0.25.0)\n",
      "Requirement already satisfied: psutil in /opt/conda/lib/python3.11/site-packages (from accelerate>=0.21.0->transformers[torch]) (5.9.5)\n",
      "Requirement already satisfied: fsspec>=2023.5.0 in /opt/conda/lib/python3.11/site-packages (from huggingface-hub<1.0,>=0.19.3->transformers[torch]) (2023.10.0)\n",
      "Requirement already satisfied: typing-extensions>=3.7.4.3 in /opt/conda/lib/python3.11/site-packages (from huggingface-hub<1.0,>=0.19.3->transformers[torch]) (4.8.0)\n",
      "Requirement already satisfied: sympy in /opt/conda/lib/python3.11/site-packages (from torch!=1.12.0,>=1.10->transformers[torch]) (1.12)\n",
      "Requirement already satisfied: networkx in /opt/conda/lib/python3.11/site-packages (from torch!=1.12.0,>=1.10->transformers[torch]) (3.0)\n",
      "Requirement already satisfied: jinja2 in /opt/conda/lib/python3.11/site-packages (from torch!=1.12.0,>=1.10->transformers[torch]) (3.1.2)\n",
      "Requirement already satisfied: triton==2.1.0 in /opt/conda/lib/python3.11/site-packages (from torch!=1.12.0,>=1.10->transformers[torch]) (2.1.0)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in /opt/conda/lib/python3.11/site-packages (from requests->transformers[torch]) (3.3.2)\n",
      "Requirement already satisfied: idna<4,>=2.5 in /opt/conda/lib/python3.11/site-packages (from requests->transformers[torch]) (3.4)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/conda/lib/python3.11/site-packages (from requests->transformers[torch]) (2.1.0)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.11/site-packages (from requests->transformers[torch]) (2023.7.22)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in /opt/conda/lib/python3.11/site-packages (from jinja2->torch!=1.12.0,>=1.10->transformers[torch]) (2.1.3)\n",
      "Requirement already satisfied: mpmath>=0.19 in /opt/conda/lib/python3.11/site-packages (from sympy->torch!=1.12.0,>=1.10->transformers[torch]) (1.3.0)\n",
      "Requirement already satisfied: datasets in /opt/conda/lib/python3.11/site-packages (2.16.1)\n",
      "Requirement already satisfied: filelock in /opt/conda/lib/python3.11/site-packages (from datasets) (3.9.0)\n",
      "Requirement already satisfied: numpy>=1.17 in /opt/conda/lib/python3.11/site-packages (from datasets) (1.26.0)\n",
      "Requirement already satisfied: pyarrow>=8.0.0 in /opt/conda/lib/python3.11/site-packages (from datasets) (14.0.2)\n",
      "Requirement already satisfied: pyarrow-hotfix in /opt/conda/lib/python3.11/site-packages (from datasets) (0.6)\n",
      "Requirement already satisfied: dill<0.3.8,>=0.3.0 in /opt/conda/lib/python3.11/site-packages (from datasets) (0.3.7)\n",
      "Requirement already satisfied: pandas in /opt/conda/lib/python3.11/site-packages (from datasets) (2.1.4)\n",
      "Requirement already satisfied: requests>=2.19.0 in /opt/conda/lib/python3.11/site-packages (from datasets) (2.31.0)\n",
      "Requirement already satisfied: tqdm>=4.62.1 in /opt/conda/lib/python3.11/site-packages (from datasets) (4.66.1)\n",
      "Requirement already satisfied: xxhash in /opt/conda/lib/python3.11/site-packages (from datasets) (3.4.1)\n",
      "Requirement already satisfied: multiprocess in /opt/conda/lib/python3.11/site-packages (from datasets) (0.70.15)\n",
      "Requirement already satisfied: fsspec<=2023.10.0,>=2023.1.0 in /opt/conda/lib/python3.11/site-packages (from fsspec[http]<=2023.10.0,>=2023.1.0->datasets) (2023.10.0)\n",
      "Requirement already satisfied: aiohttp in /opt/conda/lib/python3.11/site-packages (from datasets) (3.8.6)\n",
      "Requirement already satisfied: huggingface-hub>=0.19.4 in /opt/conda/lib/python3.11/site-packages (from datasets) (0.20.1)\n",
      "Requirement already satisfied: packaging in /opt/conda/lib/python3.11/site-packages (from datasets) (23.2)\n",
      "Requirement already satisfied: pyyaml>=5.1 in /opt/conda/lib/python3.11/site-packages (from datasets) (6.0.1)\n",
      "Requirement already satisfied: attrs>=17.3.0 in /opt/conda/lib/python3.11/site-packages (from aiohttp->datasets) (23.1.0)\n",
      "Requirement already satisfied: charset-normalizer<4.0,>=2.0 in /opt/conda/lib/python3.11/site-packages (from aiohttp->datasets) (3.3.2)\n",
      "Requirement already satisfied: multidict<7.0,>=4.5 in /opt/conda/lib/python3.11/site-packages (from aiohttp->datasets) (6.0.4)\n",
      "Requirement already satisfied: async-timeout<5.0,>=4.0.0a3 in /opt/conda/lib/python3.11/site-packages (from aiohttp->datasets) (4.0.3)\n",
      "Requirement already satisfied: yarl<2.0,>=1.0 in /opt/conda/lib/python3.11/site-packages (from aiohttp->datasets) (1.9.2)\n",
      "Requirement already satisfied: frozenlist>=1.1.1 in /opt/conda/lib/python3.11/site-packages (from aiohttp->datasets) (1.4.0)\n",
      "Requirement already satisfied: aiosignal>=1.1.2 in /opt/conda/lib/python3.11/site-packages (from aiohttp->datasets) (1.3.1)\n",
      "Requirement already satisfied: typing-extensions>=3.7.4.3 in /opt/conda/lib/python3.11/site-packages (from huggingface-hub>=0.19.4->datasets) (4.8.0)\n",
      "Requirement already satisfied: idna<4,>=2.5 in /opt/conda/lib/python3.11/site-packages (from requests>=2.19.0->datasets) (3.4)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/conda/lib/python3.11/site-packages (from requests>=2.19.0->datasets) (2.1.0)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.11/site-packages (from requests>=2.19.0->datasets) (2023.7.22)\n",
      "Requirement already satisfied: python-dateutil>=2.8.2 in /opt/conda/lib/python3.11/site-packages (from pandas->datasets) (2.8.2)\n",
      "Requirement already satisfied: pytz>=2020.1 in /opt/conda/lib/python3.11/site-packages (from pandas->datasets) (2023.3.post1)\n",
      "Requirement already satisfied: tzdata>=2022.1 in /opt/conda/lib/python3.11/site-packages (from pandas->datasets) (2023.4)\n",
      "Requirement already satisfied: six>=1.5 in /opt/conda/lib/python3.11/site-packages (from python-dateutil>=2.8.2->pandas->datasets) (1.16.0)\n",
      "Requirement already satisfied: evaluate in /opt/conda/lib/python3.11/site-packages (0.4.1)\n",
      "Requirement already satisfied: datasets>=2.0.0 in /opt/conda/lib/python3.11/site-packages (from evaluate) (2.16.1)\n",
      "Requirement already satisfied: numpy>=1.17 in /opt/conda/lib/python3.11/site-packages (from evaluate) (1.26.0)\n",
      "Requirement already satisfied: dill in /opt/conda/lib/python3.11/site-packages (from evaluate) (0.3.7)\n",
      "Requirement already satisfied: pandas in /opt/conda/lib/python3.11/site-packages (from evaluate) (2.1.4)\n",
      "Requirement already satisfied: requests>=2.19.0 in /opt/conda/lib/python3.11/site-packages (from evaluate) (2.31.0)\n",
      "Requirement already satisfied: tqdm>=4.62.1 in /opt/conda/lib/python3.11/site-packages (from evaluate) (4.66.1)\n",
      "Requirement already satisfied: xxhash in /opt/conda/lib/python3.11/site-packages (from evaluate) (3.4.1)\n",
      "Requirement already satisfied: multiprocess in /opt/conda/lib/python3.11/site-packages (from evaluate) (0.70.15)\n",
      "Requirement already satisfied: fsspec>=2021.05.0 in /opt/conda/lib/python3.11/site-packages (from fsspec[http]>=2021.05.0->evaluate) (2023.10.0)\n",
      "Requirement already satisfied: huggingface-hub>=0.7.0 in /opt/conda/lib/python3.11/site-packages (from evaluate) (0.20.1)\n",
      "Requirement already satisfied: packaging in /opt/conda/lib/python3.11/site-packages (from evaluate) (23.2)\n",
      "Requirement already satisfied: responses<0.19 in /opt/conda/lib/python3.11/site-packages (from evaluate) (0.18.0)\n",
      "Requirement already satisfied: filelock in /opt/conda/lib/python3.11/site-packages (from datasets>=2.0.0->evaluate) (3.9.0)\n",
      "Requirement already satisfied: pyarrow>=8.0.0 in /opt/conda/lib/python3.11/site-packages (from datasets>=2.0.0->evaluate) (14.0.2)\n",
      "Requirement already satisfied: pyarrow-hotfix in /opt/conda/lib/python3.11/site-packages (from datasets>=2.0.0->evaluate) (0.6)\n",
      "Requirement already satisfied: aiohttp in /opt/conda/lib/python3.11/site-packages (from datasets>=2.0.0->evaluate) (3.8.6)\n",
      "Requirement already satisfied: pyyaml>=5.1 in /opt/conda/lib/python3.11/site-packages (from datasets>=2.0.0->evaluate) (6.0.1)\n",
      "Requirement already satisfied: typing-extensions>=3.7.4.3 in /opt/conda/lib/python3.11/site-packages (from huggingface-hub>=0.7.0->evaluate) (4.8.0)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in /opt/conda/lib/python3.11/site-packages (from requests>=2.19.0->evaluate) (3.3.2)\n",
      "Requirement already satisfied: idna<4,>=2.5 in /opt/conda/lib/python3.11/site-packages (from requests>=2.19.0->evaluate) (3.4)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/conda/lib/python3.11/site-packages (from requests>=2.19.0->evaluate) (2.1.0)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.11/site-packages (from requests>=2.19.0->evaluate) (2023.7.22)\n",
      "Requirement already satisfied: python-dateutil>=2.8.2 in /opt/conda/lib/python3.11/site-packages (from pandas->evaluate) (2.8.2)\n",
      "Requirement already satisfied: pytz>=2020.1 in /opt/conda/lib/python3.11/site-packages (from pandas->evaluate) (2023.3.post1)\n",
      "Requirement already satisfied: tzdata>=2022.1 in /opt/conda/lib/python3.11/site-packages (from pandas->evaluate) (2023.4)\n",
      "Requirement already satisfied: attrs>=17.3.0 in /opt/conda/lib/python3.11/site-packages (from aiohttp->datasets>=2.0.0->evaluate) (23.1.0)\n",
      "Requirement already satisfied: multidict<7.0,>=4.5 in /opt/conda/lib/python3.11/site-packages (from aiohttp->datasets>=2.0.0->evaluate) (6.0.4)\n",
      "Requirement already satisfied: async-timeout<5.0,>=4.0.0a3 in /opt/conda/lib/python3.11/site-packages (from aiohttp->datasets>=2.0.0->evaluate) (4.0.3)\n",
      "Requirement already satisfied: yarl<2.0,>=1.0 in /opt/conda/lib/python3.11/site-packages (from aiohttp->datasets>=2.0.0->evaluate) (1.9.2)\n",
      "Requirement already satisfied: frozenlist>=1.1.1 in /opt/conda/lib/python3.11/site-packages (from aiohttp->datasets>=2.0.0->evaluate) (1.4.0)\n",
      "Requirement already satisfied: aiosignal>=1.1.2 in /opt/conda/lib/python3.11/site-packages (from aiohttp->datasets>=2.0.0->evaluate) (1.3.1)\n",
      "Requirement already satisfied: six>=1.5 in /opt/conda/lib/python3.11/site-packages (from python-dateutil>=2.8.2->pandas->evaluate) (1.16.0)\n",
      "Requirement already satisfied: scikit-learn in /opt/conda/lib/python3.11/site-packages (1.3.2)\n",
      "Requirement already satisfied: numpy<2.0,>=1.17.3 in /opt/conda/lib/python3.11/site-packages (from scikit-learn) (1.26.0)\n",
      "Requirement already satisfied: scipy>=1.5.0 in /opt/conda/lib/python3.11/site-packages (from scikit-learn) (1.11.4)\n",
      "Requirement already satisfied: joblib>=1.1.1 in /opt/conda/lib/python3.11/site-packages (from scikit-learn) (1.3.2)\n",
      "Requirement already satisfied: threadpoolctl>=2.0.0 in /opt/conda/lib/python3.11/site-packages (from scikit-learn) (3.2.0)\n",
      "Requirement already satisfied: wandb in /opt/conda/lib/python3.11/site-packages (0.16.1)\n",
      "Requirement already satisfied: Click!=8.0.0,>=7.1 in /opt/conda/lib/python3.11/site-packages (from wandb) (8.1.7)\n",
      "Requirement already satisfied: GitPython!=3.1.29,>=1.0.0 in /opt/conda/lib/python3.11/site-packages (from wandb) (3.1.40)\n",
      "Requirement already satisfied: requests<3,>=2.0.0 in /opt/conda/lib/python3.11/site-packages (from wandb) (2.31.0)\n",
      "Requirement already satisfied: psutil>=5.0.0 in /opt/conda/lib/python3.11/site-packages (from wandb) (5.9.5)\n",
      "Requirement already satisfied: sentry-sdk>=1.0.0 in /opt/conda/lib/python3.11/site-packages (from wandb) (1.39.1)\n",
      "Requirement already satisfied: docker-pycreds>=0.4.0 in /opt/conda/lib/python3.11/site-packages (from wandb) (0.4.0)\n",
      "Requirement already satisfied: PyYAML in /opt/conda/lib/python3.11/site-packages (from wandb) (6.0.1)\n",
      "Requirement already satisfied: setproctitle in /opt/conda/lib/python3.11/site-packages (from wandb) (1.3.3)\n",
      "Requirement already satisfied: setuptools in /opt/conda/lib/python3.11/site-packages (from wandb) (68.2.2)\n",
      "Requirement already satisfied: appdirs>=1.4.3 in /opt/conda/lib/python3.11/site-packages (from wandb) (1.4.4)\n",
      "Requirement already satisfied: protobuf!=4.21.0,<5,>=3.19.0 in /opt/conda/lib/python3.11/site-packages (from wandb) (4.25.1)\n",
      "Requirement already satisfied: six>=1.4.0 in /opt/conda/lib/python3.11/site-packages (from docker-pycreds>=0.4.0->wandb) (1.16.0)\n",
      "Requirement already satisfied: gitdb<5,>=4.0.1 in /opt/conda/lib/python3.11/site-packages (from GitPython!=3.1.29,>=1.0.0->wandb) (4.0.11)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in /opt/conda/lib/python3.11/site-packages (from requests<3,>=2.0.0->wandb) (3.3.2)\n",
      "Requirement already satisfied: idna<4,>=2.5 in /opt/conda/lib/python3.11/site-packages (from requests<3,>=2.0.0->wandb) (3.4)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/conda/lib/python3.11/site-packages (from requests<3,>=2.0.0->wandb) (2.1.0)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.11/site-packages (from requests<3,>=2.0.0->wandb) (2023.7.22)\n",
      "Requirement already satisfied: smmap<6,>=3.0.1 in /opt/conda/lib/python3.11/site-packages (from gitdb<5,>=4.0.1->GitPython!=3.1.29,>=1.0.0->wandb) (5.0.1)\n"
     ]
    }
   ],
   "source": [
    "! pip install transformers[torch]\n",
    "! pip install datasets\n",
    "! pip install evaluate\n",
    "! pip install scikit-learn\n",
    "! pip install wandb\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "81ba13c7-c749-4988-937e-83c170dce08f",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/opt/conda/lib/python3.11/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html\n",
      "  from .autonotebook import tqdm as notebook_tqdm\n"
     ]
    }
   ],
   "source": [
    "import transformers\n",
    "from datasets import load_dataset\n",
    "from transformers import AutoTokenizer\n",
    "from transformers import TrainingArguments, Trainer\n",
    "import wandb\n",
    "\n",
    "import numpy as np\n",
    "import evaluate\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "90d4da41-b471-44da-86e5-2df89ef75b63",
   "metadata": {},
   "source": [
    "# Login to Weights and Biases\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "3c7258a6-ba86-46b4-920b-8eae840d12f8",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[34m\u001b[1mwandb\u001b[0m: Currently logged in as: \u001b[33mddahlmeier\u001b[0m (\u001b[33msutd-mlops\u001b[0m). Use \u001b[1m`wandb login --relogin`\u001b[0m to force relogin\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "wandb.login()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "3cbdacef-88c6-4163-bf8f-7a91e766bf6d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "Tracking run with wandb version 0.16.1"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Run data is saved locally in <code>/home/jovyan/wandb/run-20231231_075939-4kmu4dyq</code>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Syncing run <strong><a href='https://wandb.ai/sutd-mlops/sutd-mlops-project/runs/4kmu4dyq' target=\"_blank\">comic-night-6</a></strong> to <a href='https://wandb.ai/sutd-mlops/sutd-mlops-project' target=\"_blank\">Weights & Biases</a> (<a href='https://wandb.me/run' target=\"_blank\">docs</a>)<br/>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View project at <a href='https://wandb.ai/sutd-mlops/sutd-mlops-project' target=\"_blank\">https://wandb.ai/sutd-mlops/sutd-mlops-project</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View run at <a href='https://wandb.ai/sutd-mlops/sutd-mlops-project/runs/4kmu4dyq' target=\"_blank\">https://wandb.ai/sutd-mlops/sutd-mlops-project/runs/4kmu4dyq</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "env: WANDB_WATCH=all\n"
     ]
    }
   ],
   "source": [
    "wandb.init(project=\"sutd-mlops-project\")\n",
    "# Optional: log both gradients and parameters\n",
    "%env WANDB_WATCH=all"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "f188f12a-8acd-4aed-9713-9df9c6a5629a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'text': 'the rock is destined to be the 21st century\\'s new \" conan \" and that he\\'s going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .',\n",
       " 'label': 1}"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset = load_dataset(\"rotten_tomatoes\")\n",
    "dataset[\"train\"][0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "1373b958-c7e9-4dc0-aacd-b0277c5f8f54",
   "metadata": {},
   "outputs": [],
   "source": [
    "model_name = \"distilbert-base-uncased\"\n",
    "tokenizer = AutoTokenizer.from_pretrained(model_name)\n",
    "\n",
    "def tokenize_function(examples):\n",
    "    return tokenizer(examples[\"text\"], padding=\"max_length\", truncation=True)\n",
    "\n",
    "tokenized_datasets = dataset.map(tokenize_function, batched=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "fe765924-396d-4055-ab91-fe8e6d1229ab",
   "metadata": {},
   "outputs": [],
   "source": [
    "small_train_dataset = tokenized_datasets[\"train\"].shuffle(seed=42).select(range(1000))\n",
    "small_eval_dataset = tokenized_datasets[\"test\"].shuffle(seed=42).select(range(1000))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "0caefc42-7124-4108-b39f-6e5d57d7d1b6",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['pre_classifier.weight', 'classifier.weight', 'classifier.bias', 'pre_classifier.bias']\n",
      "You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.\n"
     ]
    }
   ],
   "source": [
    "from transformers import AutoModelForSequenceClassification\n",
    "\n",
    "model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "664c8e19-1fdb-48cb-b736-cd900e47137c",
   "metadata": {},
   "outputs": [],
   "source": [
    "training_args = TrainingArguments(output_dir=\"test_trainer\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "4b4a7a7e-4bcb-42f7-9a43-45e2199360fc",
   "metadata": {},
   "outputs": [],
   "source": [
    "metric = evaluate.load(\"accuracy\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "7716db15-d0f0-4bd4-90d1-8881454784ce",
   "metadata": {},
   "outputs": [],
   "source": [
    "def compute_metrics(eval_pred):\n",
    "    logits, labels = eval_pred\n",
    "    predictions = np.argmax(logits, axis=-1)\n",
    "    return metric.compute(predictions=predictions, references=labels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "1a3d1697-4a2a-4679-8efe-cb6107fdc1de",
   "metadata": {},
   "outputs": [],
   "source": [
    "training_args = TrainingArguments(output_dir=\"test_trainer\", evaluation_strategy=\"epoch\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "b961a807-9b5e-4425-8be9-51716a04e2c4",
   "metadata": {},
   "outputs": [],
   "source": [
    "trainer = Trainer(\n",
    "    model=model,\n",
    "    args=training_args,\n",
    "    train_dataset=small_train_dataset,\n",
    "    eval_dataset=small_eval_dataset,\n",
    "    compute_metrics=compute_metrics,\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "3e77751c-17b5-4be3-a57e-5ffe45a2014b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "\n",
       "    <div>\n",
       "      \n",
       "      <progress value='375' max='375' style='width:300px; height:20px; vertical-align: middle;'></progress>\n",
       "      [375/375 01:05, Epoch 3/3]\n",
       "    </div>\n",
       "    <table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       " <tr style=\"text-align: left;\">\n",
       "      <th>Epoch</th>\n",
       "      <th>Training Loss</th>\n",
       "      <th>Validation Loss</th>\n",
       "      <th>Accuracy</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>No log</td>\n",
       "      <td>0.580372</td>\n",
       "      <td>0.738000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>No log</td>\n",
       "      <td>0.771544</td>\n",
       "      <td>0.781000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>No log</td>\n",
       "      <td>0.849353</td>\n",
       "      <td>0.797000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table><p>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "TrainOutput(global_step=375, training_loss=0.2987349853515625, metrics={'train_runtime': 66.3165, 'train_samples_per_second': 45.238, 'train_steps_per_second': 5.655, 'total_flos': 397402195968000.0, 'train_loss': 0.2987349853515625, 'epoch': 3.0})"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "trainer.train()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8d5163d3-91c1-47fc-be06-68a263d660c9",
   "metadata": {},
   "source": [
    "# What to try next\n",
    "\n",
    "- train and evaluate with the complete training and test dataset instead of a sample\n",
    "- experiment with different training parameters (number of epochs, optimizers, batch size, learning rate schedule, ...)\n",
    "- compare DistilBERT vs the full BERT model\n",
    "- compare the results with the scikit model from the previous notebook. What is the cost-benefit trade off between deep learning and traditional ML?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "def5476d-8c3c-4eec-aea2-d0017d99ccff",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>\n",
       "    table.wandb td:nth-child(1) { padding: 0 10px; text-align: left ; width: auto;} td:nth-child(2) {text-align: left ; width: 100%}\n",
       "    .wandb-row { display: flex; flex-direction: row; flex-wrap: wrap; justify-content: flex-start; width: 100% }\n",
       "    .wandb-col { display: flex; flex-direction: column; flex-basis: 100%; flex: 1; padding: 10px; }\n",
       "    </style>\n",
       "<div class=\"wandb-row\"><div class=\"wandb-col\"><h3>Run history:</h3><br/><table class=\"wandb\"><tr><td>eval/accuracy</td><td>▁▆█</td></tr><tr><td>eval/loss</td><td>▁▆█</td></tr><tr><td>eval/runtime</td><td>▁█▂</td></tr><tr><td>eval/samples_per_second</td><td>█▁▇</td></tr><tr><td>eval/steps_per_second</td><td>█▁▇</td></tr><tr><td>train/epoch</td><td>▁▅██</td></tr><tr><td>train/global_step</td><td>▁▅██</td></tr><tr><td>train/total_flos</td><td>▁</td></tr><tr><td>train/train_loss</td><td>▁</td></tr><tr><td>train/train_runtime</td><td>▁</td></tr><tr><td>train/train_samples_per_second</td><td>▁</td></tr><tr><td>train/train_steps_per_second</td><td>▁</td></tr></table><br/></div><div class=\"wandb-col\"><h3>Run summary:</h3><br/><table class=\"wandb\"><tr><td>eval/accuracy</td><td>0.797</td></tr><tr><td>eval/loss</td><td>0.84935</td></tr><tr><td>eval/runtime</td><td>5.713</td></tr><tr><td>eval/samples_per_second</td><td>175.04</td></tr><tr><td>eval/steps_per_second</td><td>21.88</td></tr><tr><td>train/epoch</td><td>3.0</td></tr><tr><td>train/global_step</td><td>375</td></tr><tr><td>train/total_flos</td><td>397402195968000.0</td></tr><tr><td>train/train_loss</td><td>0.29873</td></tr><tr><td>train/train_runtime</td><td>66.3165</td></tr><tr><td>train/train_samples_per_second</td><td>45.238</td></tr><tr><td>train/train_steps_per_second</td><td>5.655</td></tr></table><br/></div></div>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View run <strong style=\"color:#cdcd00\">comic-night-6</strong> at: <a href='https://wandb.ai/sutd-mlops/sutd-mlops-project/runs/4kmu4dyq' target=\"_blank\">https://wandb.ai/sutd-mlops/sutd-mlops-project/runs/4kmu4dyq</a><br/>Synced 6 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Find logs at: <code>./wandb/run-20231231_075939-4kmu4dyq/logs</code>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "wandb.finish()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4243813d-eec3-440a-9b4f-12658bd14711",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fa62daf2-ae0a-4d57-b197-427255c474c5",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
