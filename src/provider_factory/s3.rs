use anyhow::{Context, Result};

use super::{ProviderFactoryArgs, owned};
use crate::storage::s3::{S3ConfigInput, S3Uploader, s3_config_from_input};

pub(super) async fn build_uploader(
    args: &ProviderFactoryArgs<'_>,
    default_prefix: &str,
    context_label: &str,
) -> Result<S3Uploader> {
    let bucket = args
        .s3_bucket
        .with_context(|| format!("--s3-bucket or S3_BUCKET is required for {context_label}"))?;
    let region = owned(args.s3_region)
        .or_else(|| std::env::var("AWS_REGION").ok())
        .with_context(|| {
            format!("--s3-region, S3_REGION, or AWS_REGION is required for {context_label}")
        })?;
    let access_key_id = owned(args.s3_access_key_id)
        .or_else(|| std::env::var("AWS_ACCESS_KEY_ID").ok())
        .with_context(|| {
            format!(
                "--s3-access-key-id, S3_ACCESS_KEY_ID, or AWS_ACCESS_KEY_ID is required for {context_label}"
            )
        })?;
    let secret_access_key = owned(args.s3_secret_access_key)
        .or_else(|| std::env::var("AWS_SECRET_ACCESS_KEY").ok())
        .with_context(|| {
            format!(
                "--s3-secret-access-key, S3_SECRET_ACCESS_KEY, or AWS_SECRET_ACCESS_KEY is required for {context_label}"
            )
        })?;
    let config = s3_config_from_input(S3ConfigInput {
        bucket: bucket.to_string(),
        region,
        endpoint_url: owned(args.s3_endpoint_url),
        access_key_id,
        secret_access_key,
        session_token: owned(args.s3_session_token)
            .or_else(|| std::env::var("AWS_SESSION_TOKEN").ok()),
        prefix: owned(args.s3_prefix).or_else(|| Some(default_prefix.to_string())),
        presign_expires_secs: args.s3_presign_expires_secs,
        force_path_style: args.s3_force_path_style,
    })?;
    S3Uploader::new(config).await
}
