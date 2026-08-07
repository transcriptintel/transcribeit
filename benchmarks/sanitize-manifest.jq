def sensitive_key:
  test("^(file|file_name|file_uri|url|uri|request_id|provider_request_id|text|transcript|segments|words|cache_index_path)$"; "i");

walk(
  if type == "object" then
    with_entries(select((.key | sensitive_key) | not))
  else
    .
  end
)
