#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{Error as E, Result};
use candle_core::Tensor;
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::{PaddingParams, Tokenizer};

static DEVICE: candle_core::Device = candle_core::Device::Cpu;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    prompt: Option<String>,

    #[arg(long, default_value = "false")]
    normalize_embeddings: bool,
}


fn build_model_and_tokenizer() -> Result<(BertModel, Tokenizer)> {
    let device = &DEVICE;
    let model_id = "sentence-transformers/all-MiniLM-L6-v2".to_string();
    let revision = "refs/pr/21".to_string();

    let repo = Repo::with_revision(model_id, RepoType::Model, revision);
    let (config_filename, tokenizer_filename, weights_filename) = {
        let api = Api::new()?;
        let api = api.repo(repo);
        let config = api.get("config.json")?;
        let tokenizer = api.get("tokenizer.json")?;
        let weights = api.get("model.safetensors")?;
        (config, tokenizer, weights)
    };
    let config = std::fs::read_to_string(config_filename)?;
    let config: Config = serde_json::from_str(&config)?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };
    let model = BertModel::load(vb, &config)?;
    Ok((model, tokenizer))
}

fn main() -> Result<()> {
    let (model, mut tokenizer) = build_model_and_tokenizer().unwrap();
    let device = &DEVICE;
    let args = Args::parse();
    if let Some(prompt) = args.prompt {
        println!("prompt: {prompt}");
        let tokenizer = tokenizer
            .with_padding(None)
            .with_truncation(None)
            .map_err(E::msg)?;
        let tokens = tokenizer
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
        let token_type_ids = token_ids.zeros_like()?;
        // let start = std::time::Instant::now();
        let ys = model.forward(&token_ids, &token_type_ids, None)?;
        let embeddings = if args.normalize_embeddings {
        normalize_l2(&ys)?
        } else {
            ys
        };
        println!("{embeddings}");
        // println!("Took {:?}", start.elapsed());

    } else {
        let sentences = [
            "The cat sits outside",
            "A man is playing guitar",
            "I love pasta",
            "The new movie is awesome",
            "The cat plays in the garden",
            "A woman watches TV",
            "The new movie is so great",
            "Do you like pizza?",
        ];
        let n_sentences = sentences.len();
        if let Some(pp) = tokenizer.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest
        } else {
            let pp = PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            tokenizer.with_padding(Some(pp));
        }
        let tokens = tokenizer
            .encode_batch(sentences.to_vec(), true)
            .map_err(E::msg)?;
        let token_ids = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_ids().to_vec();
                Ok(Tensor::new(tokens.as_slice(), device)?)
            })
            .collect::<Result<Vec<_>>>()?;
        let attention_mask = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_attention_mask().to_vec();
                Ok(Tensor::new(tokens.as_slice(), device)?)
            })
            .collect::<Result<Vec<_>>>()?;

        let token_ids = Tensor::stack(&token_ids, 0)?;
        let attention_mask = Tensor::stack(&attention_mask, 0)?;
        let token_type_ids = token_ids.zeros_like()?;
        println!("running inference on batch {:?}", token_ids.shape());
        let embeddings = model.forward(&token_ids, &token_type_ids, Some(&attention_mask))?;
        println!("generated embeddings {:?}", embeddings.shape());
        // Apply some avg-pooling by taking the mean embedding value for all tokens (including padding)
        let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
        let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
        let embeddings = if args.normalize_embeddings {
            println!("normalized embeddings {:?}", embeddings.shape());
            normalize_l2(&embeddings)?
        } else {
            embeddings
        };
        println!("pooled embeddings {:?}", embeddings.shape());

        let mut similarities = vec![];
        for i in 0..n_sentences {
            let e_i = embeddings.get(i)?;
            for j in (i + 1)..n_sentences {
                let e_j = embeddings.get(j)?;
                let sum_ij = (&e_i * &e_j)?.sum_all()?.to_scalar::<f32>()?;
                let sum_i2 = (&e_i * &e_i)?.sum_all()?.to_scalar::<f32>()?;
                let sum_j2 = (&e_j * &e_j)?.sum_all()?.to_scalar::<f32>()?;
                let cosine_similarity = sum_ij / (sum_i2 * sum_j2).sqrt();
                similarities.push((cosine_similarity, i, j))
            }
        }
        similarities.sort_by(|u, v| v.0.total_cmp(&u.0));
        for &(score, i, j) in similarities[..5].iter() {
            println!("score: {score:.2} '{}' '{}'", sentences[i], sentences[j])
        }
    }
    Ok(())
}

pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}
