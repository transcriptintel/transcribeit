use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result};
use aws_config::{BehaviorVersion, Region};
use aws_credential_types::Credentials;
use aws_credential_types::provider::SharedCredentialsProvider;
use aws_sdk_s3::Client;
use aws_sdk_s3::config::Builder;
use aws_sdk_s3::presigning::PresigningConfig;
use aws_sdk_s3::primitives::ByteStream;
use serde_json::{Value, json};
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct S3Config {
    pub bucket: String,
    pub region: String,
    pub endpoint_url: Option<String>,
    pub access_key_id: String,
    pub secret_access_key: String,
    pub session_token: Option<String>,
    pub prefix: String,
    pub presign_expires: Duration,
    pub force_path_style: bool,
}

pub struct S3ConfigInput {
    pub bucket: String,
    pub region: String,
    pub endpoint_url: Option<String>,
    pub access_key_id: String,
    pub secret_access_key: String,
    pub session_token: Option<String>,
    pub prefix: Option<String>,
    pub presign_expires_secs: u64,
    pub force_path_style: bool,
}

#[derive(Clone)]
pub struct S3Uploader {
    client: Client,
    bucket: String,
    prefix: String,
    presign_expires: Duration,
}

#[derive(Debug, Clone)]
pub struct S3UploadedObject {
    pub url: String,
    pub bucket: String,
    pub key: String,
}

#[derive(Debug, Clone)]
pub struct S3CleanupResult {
    pub attempted: bool,
    pub deleted: bool,
    pub bucket: String,
    pub key: String,
    pub error: Option<String>,
}

impl S3CleanupResult {
    pub fn skipped(upload: &S3UploadedObject) -> Self {
        Self {
            attempted: false,
            deleted: false,
            bucket: upload.bucket.clone(),
            key: upload.key.clone(),
            error: None,
        }
    }

    pub fn to_metadata(&self) -> Value {
        json!({
            "attempted": self.attempted,
            "deleted": self.deleted,
            "bucket": self.bucket,
            "key": self.key,
            "error": self.error,
        })
    }
}

impl S3Uploader {
    pub async fn new(config: S3Config) -> Result<Self> {
        let credentials = Credentials::new(
            config.access_key_id,
            config.secret_access_key,
            config.session_token,
            None,
            "transcribeit-s3",
        );

        let mut loader = aws_config::defaults(BehaviorVersion::latest())
            .region(Region::new(config.region))
            .credentials_provider(SharedCredentialsProvider::new(credentials));

        if let Some(endpoint_url) = config.endpoint_url {
            loader = loader.endpoint_url(endpoint_url);
        }

        let shared_config = loader.load().await;
        let s3_config = Builder::from(&shared_config)
            .force_path_style(config.force_path_style)
            .build();

        Ok(Self {
            client: Client::from_conf(s3_config),
            bucket: config.bucket,
            prefix: config.prefix.trim_matches('/').to_string(),
            presign_expires: config.presign_expires,
        })
    }

    pub async fn upload_and_presign_object(&self, path: &Path) -> Result<S3UploadedObject> {
        let key = self.object_key(path);
        let body = ByteStream::from_path(path)
            .await
            .with_context(|| format!("Failed to read upload file: {}", path.display()))?;
        let content_type = content_type(path);

        self.client
            .put_object()
            .bucket(&self.bucket)
            .key(&key)
            .body(body)
            .content_type(content_type)
            .send()
            .await
            .with_context(|| format!("Failed to upload {} to S3", path.display()))?;

        let presigned = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(&key)
            .presigned(PresigningConfig::expires_in(self.presign_expires)?)
            .await
            .with_context(|| {
                format!("Failed to presign S3 object: s3://{}/{}", self.bucket, key)
            })?;

        Ok(S3UploadedObject {
            url: presigned.uri().to_string(),
            bucket: self.bucket.clone(),
            key,
        })
    }

    pub async fn cleanup_uploaded_object(&self, upload: &S3UploadedObject) -> S3CleanupResult {
        let delete_result = self
            .client
            .delete_object()
            .bucket(&upload.bucket)
            .key(&upload.key)
            .send()
            .await;

        let deleted = delete_result.is_ok();
        S3CleanupResult {
            attempted: true,
            deleted,
            bucket: upload.bucket.clone(),
            key: upload.key.clone(),
            error: delete_result.err().map(|err| err.to_string()),
        }
    }

    fn object_key(&self, path: &Path) -> String {
        let file_name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .replace('/', "_");
        let object = format!("{}-{}", Uuid::new_v4(), file_name);

        if self.prefix.is_empty() {
            object
        } else {
            format!("{}/{}", self.prefix, object)
        }
    }
}

fn content_type(path: &Path) -> &'static str {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) if ext.eq_ignore_ascii_case("mp3") => "audio/mpeg",
        Some(ext) if ext.eq_ignore_ascii_case("mp4") => "video/mp4",
        Some(ext) if ext.eq_ignore_ascii_case("m4a") => "audio/mp4",
        Some(ext) if ext.eq_ignore_ascii_case("wav") => "audio/wav",
        _ => "application/octet-stream",
    }
}

pub fn s3_config_from_input(input: S3ConfigInput) -> Result<S3Config> {
    let region = fallback_env(input.region, "AWS_REGION");
    let access_key_id = fallback_env(input.access_key_id, "AWS_ACCESS_KEY_ID");
    let secret_access_key = fallback_env(input.secret_access_key, "AWS_SECRET_ACCESS_KEY");
    let session_token = input
        .session_token
        .or_else(|| std::env::var("AWS_SESSION_TOKEN").ok());

    anyhow::ensure!(
        !input.bucket.trim().is_empty(),
        "S3 bucket must not be empty"
    );
    anyhow::ensure!(!region.trim().is_empty(), "S3 region must not be empty");
    anyhow::ensure!(
        !access_key_id.trim().is_empty(),
        "S3 access key ID must not be empty"
    );
    anyhow::ensure!(
        !secret_access_key.trim().is_empty(),
        "S3 secret access key must not be empty"
    );

    Ok(S3Config {
        bucket: input.bucket,
        region,
        endpoint_url: input.endpoint_url,
        access_key_id,
        secret_access_key,
        session_token,
        prefix: input
            .prefix
            .unwrap_or_else(|| "transcribeit/qwen-filetrans".into()),
        presign_expires: Duration::from_secs(input.presign_expires_secs.max(300)),
        force_path_style: input.force_path_style,
    })
}

fn fallback_env(value: String, env_key: &str) -> String {
    if value.trim().is_empty() {
        std::env::var(env_key).unwrap_or(value)
    } else {
        value
    }
}
