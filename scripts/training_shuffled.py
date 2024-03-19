from argparse import ArgumentParser
from peft import LoraConfig, TaskType
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from trl import AutoModelForCausalLMWithValueHead
import os

# EXAMPLE: python training_shuffled.py --model "AdaptLLM/medicine-chat" --dataset "../data/datasetTrainShuffled" --output "../models/medicine-chat-Erik-wandb" --project "PREPARE"
def main(hparams):
    model_id = hparams.model
    dataset_path = hparams.dataset
    save_model_path = hparams.output
    os.environ["WANDB_PROJECT"] = hparams.project

    access_token = "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"

    # TODO: check the lora config parameters, what do they mean and what they impact
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=64, lora_alpha=32, lora_dropout=0.1)


    #model_id = 'AdaptLLM/medicine-chat'# go for a smaller model if you dont have the VRAM
    model = AutoModelForCausalLM.from_pretrained(model_id, token=access_token, device_map="auto", load_in_4bit=True) # , peft_config=peft_config
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token


    #hg_dataset = load_from_disk("../data/dataset")
    hg_dataset = load_from_disk(dataset_path)
    #hg_dataset = hg_dataset.train_test_split(train_size=0.9)

    hg_datasetTEST = load_from_disk("../data/datasetTestShuffledEntire")


    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['input'])):
            text = f"### Your goal is to extract structured information from the user's input that matches json format. When extracting information please make sure it matches the type information exactly. Extract the following entities: {example['keys'][i]} Do not add any extra entities.\n\n### Input:\n{example['input'][i]}\n\n ### Output:\n{example['output'][i]}"
            #print(text)
            output_texts.append(text)
        return output_texts

    response_template = "### Output:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    # TODO: play around with the traning arguments, specifically the optimizer ones (learning rate, batch size, epochs)
    args = TrainingArguments(
        report_to="wandb", # enable logging to W&B
        output_dir="../models/AdaptLLM/medicine-chat-Erikk",
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=10,
        gradient_checkpointing=True,
        per_device_train_batch_size = 2,
        per_device_eval_batch_size = 2,
        do_eval=True,
        eval_steps=10,
        evaluation_strategy = "steps",
        #save_strategy="epoch",
    )

    # TODO: check the SFTTrainer parameters and how to train the model
    # Default optimizer is AdamW because its default in transformers.Trainer. SFTTrainer is a wraper around transformers.Trainer and inherits all of its attributes and methods.
    # To update weights every n batch iterations use gradient_accumulation_steps! 
    trainer = SFTTrainer(
        model,
        train_dataset=hg_dataset,
        #eval_dataset=hg_dataset["test"],
        eval_dataset=hg_datasetTEST,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        max_seq_length=2048,
        args=args,
        peft_config=peft_config,
    )

    trainer.train()

    # ! replaced approach for saving the models
    #model.save_pretrained("../models/AdaptLLM/medicine-chat-Erik")
    #trainer.save_model("../models/AdaptLLM/medicine-chat-Erik")
    trainer.save_model(save_model_path)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default=None) #--> model_id (hugging face)
    parser.add_argument("--dataset", type=str, default=None) #--> path to training dataset
    parser.add_argument("--output", type=str, default=None) #--> path for saving the model
    parser.add_argument("--project", type=str, default=None) #--> path for saving the model
    args = parser.parse_args()

    main(args)
