use std::path::PathBuf;

use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor};
use candle_nn::ops::softmax;
use candle_nn::VarBuilder;
use candle_transformers::models::xlm_roberta::{Config, XLMRobertaForSequenceClassification};
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::{PaddingParams, Tokenizer};

static DEVICE: candle_core::Device = candle_core::Device::Cpu;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// The model to use
    #[arg(long, default_value = "s-nlp/xlmr_formality_classifier")]
    model_id: String,

    #[arg(long, default_value = "main")]
    revision: String,

    /// Path to the tokenizer file.
    #[arg(long)]
    tokenizer_file: Option<String>,

    /// Path to the weight files.
    #[arg(long)]
    weight_files: Option<String>,

    /// Path to the config file.
    #[arg(long)]
    config_file: Option<String>,

    /// When set, classify text formality for this prompt.
    #[arg(long)]
    prompt: Option<String>,
}

fn build_model_and_tokenizer(
    args: &Args,
) -> Result<(XLMRobertaForSequenceClassification, Tokenizer)> {
    let device = if args.cpu {
        &DEVICE
    } else {
        #[cfg(feature = "cuda")]
        let device = candle_core::Device::new_cuda(0)?;
        #[cfg(not(feature = "cuda"))]
        let device = &DEVICE;
        device
    };

    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        args.model_id.clone(),
        RepoType::Model,
        args.revision.clone(),
    ));

    let tokenizer_filename = match &args.tokenizer_file {
        Some(file) => PathBuf::from(file),
        None => repo.get("tokenizer.json")?,
    };

    let config_filename = match &args.config_file {
        Some(file) => PathBuf::from(file),
        None => repo.get("config.json")?,
    };

    let weights_filename = match &args.weight_files {
        Some(files) => PathBuf::from(files),
        None => match repo.get("model.safetensors") {
            Ok(safetensors) => safetensors,
            Err(_) => match repo.get("pytorch_model.bin") {
                Ok(pytorch_model) => pytorch_model,
                Err(e) => {
                    return Err(anyhow::Error::msg(format!("Model weights not found. The weights should either be a `model.safetensors` or `pytorch_model.bin` file. Error: {}", e)));
                }
            },
        },
    };

    println!("Rust weights filename: {:?}", &weights_filename);
    

    let config = std::fs::read_to_string(config_filename)?;
    let config: Config = serde_json::from_str(&config)?;

    println!("Config:");
    println!("  hidden_size: {}", config.hidden_size);
    println!("  layer_norm_eps: {}", config.layer_norm_eps);
    println!("  attention_probs_dropout_prob: {}", config.attention_probs_dropout_prob);
    println!("  hidden_dropout_prob: {}", config.hidden_dropout_prob);
    println!("  num_attention_heads: {}", config.num_attention_heads);
    println!("  position_embedding_type: {:?}", config.position_embedding_type);
    println!("  intermediate_size: {}", config.intermediate_size);
    println!("  hidden_act: {:?}", config.hidden_act);
    println!("  num_hidden_layers: {}", config.num_hidden_layers);
    println!("  vocab_size: {}", config.vocab_size);
    println!("  max_position_embeddings: {}", config.max_position_embeddings);
    println!("  type_vocab_size: {}", config.type_vocab_size);
    println!("  pad_token_id: {}", config.pad_token_id);

    let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let vb = if weights_filename.ends_with("model.safetensors") {
        println!("Loading weights from safetensors");
        unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[weights_filename],
                candle_core::DType::F16,
                device,
            )?
        }

    } else {
        println!("Loading weights from pytorch_model.bin");
        VarBuilder::from_pth(&weights_filename, candle_core::DType::F16, device)?
    };

    tokenizer
        .with_padding(Some(PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            pad_id: config.pad_token_id,
            ..Default::default()
        }))
        .with_truncation(None)
        .map_err(E::msg)?;

    let model = XLMRobertaForSequenceClassification::new(2, &config, vb)?;
    Ok((model, tokenizer))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = if args.cpu {
        &DEVICE
    } else {
        #[cfg(feature = "cuda")]
        let device = candle_core::Device::new_cuda(0)?;
        #[cfg(not(feature = "cuda"))]
        let device = &DEVICE;
        device
    };

    let (model, tokenizer) = build_model_and_tokenizer(&args)?;

    if let Some(prompt) = args.prompt {
        println!("Analyzing formality of: {}", prompt);
        let sentences = vec![prompt];
        process_texts(&sentences, &model, &tokenizer, device)?;
    } else {
        let sentences = vec![
            "I like you. I love you".to_string(),
            "Hey, what's up?".to_string(),
            "Siema, co porabiasz?".to_string(),
            "I feel deep regret and sadness about the situation in international politics."
                .to_string(),
        ];
        println!("Analyzing formality");
        process_texts(&sentences, &model, &tokenizer, device)?;
    }

    Ok(())
}

fn process_texts(
    sentences: &[String],
    model: &XLMRobertaForSequenceClassification,
    tokenizer: &Tokenizer,
    device: &Device,
) -> Result<()> {
    // let inference_time = std::time::Instant::now();

    let tokens = tokenizer
        .encode_batch(sentences.to_vec(), true)
        .map_err(E::msg)?;

    let token_ids = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_ids().to_vec();
            Tensor::new(tokens.as_slice(), device)
        })
        .collect::<candle_core::Result<Vec<_>>>()?;

    let input_ids = Tensor::stack(&token_ids, 0)?;

    let attention_mask = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_attention_mask().to_vec();
            Tensor::new(tokens.as_slice(), device)
        })
        .collect::<candle_core::Result<Vec<_>>>()?;

    let attention_mask = Tensor::stack(&attention_mask, 0)?;

    let token_type_ids = Tensor::zeros(input_ids.dims(), input_ids.dtype(), device)?;

    print_tokenization_details(tokenizer, sentences, device)?;

    let logits = model
        .forward(&input_ids, &attention_mask, &token_type_ids)?
        .to_dtype(candle_core::DType::F32)?;

    // println!("Inference completed in {:?}", inference_time.elapsed());
    println!("Candle logits: {:?}", logits.to_vec2::<f32>()?);

    let predictions = logits.argmax(1)?.to_vec1::<u32>()?;
    
    println!("Predictions: {:?}", predictions);

    let probabilities = softmax(&logits, 1)?;
    let probs_vec = probabilities.to_vec2::<f32>()?;

    println!("\nFormality Scores:");
    println!("{:-<80}", "");
    for (i, (text, probs)) in sentences.iter().zip(probs_vec.iter()).enumerate() {
        println!("Text {}: \"{}\"", i + 1, text);
        println!("  formal: {:.4}", probs[0]);
        println!("  informal: {:.4}", probs[1]);
        println!();
    }
    println!("{:-<80}", "");

    Ok(())
}

fn print_tokenization_details(
    tokenizer: &Tokenizer,
    sentences: &[String],
    device: &Device,
) -> Result<()> {
    println!("========== TOKENIZATION INFO ==========");

    for (i, text) in sentences.iter().enumerate() {
        let encoding = tokenizer.encode(text.clone(), true).map_err(E::msg)?;
        let tokens = encoding.get_tokens().to_vec();
        let token_ids = encoding.get_ids().to_vec();

        println!("\nText {}: \"{}\"", i + 1, text);
        println!("Tokens: {:?}", tokens);
        println!("Token IDs: {:?}", token_ids);
        println!("Token ID to Token Mapping:");

        let bos_token_id = tokenizer.token_to_id("<s>").unwrap_or(0);
        let eos_token_id = tokenizer.token_to_id("</s>").unwrap_or(0);

        for (idx, &id) in token_ids.iter().enumerate() {
            if id != bos_token_id && id != eos_token_id {
                let unknown = "[UNKNOWN]".to_string();
                let token = tokens.get(idx).unwrap_or(&unknown);
                println!("  {} → {}", id, token);
            }
        }
        println!("{}", "-".repeat(40));
    }

    println!("=======================================\n");
    Ok(())
}
