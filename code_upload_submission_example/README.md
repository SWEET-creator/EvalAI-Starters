# Code Upload Submission Example

Build and push this image, then submit a JSON file containing the pushed image URI:

```json
{
  "submitted_image_uri": "registry.example.com/anime-colorization-submission:latest"
}
```

At evaluation time EvalAI mounts the static dataset at `/dataset` and expects the
container to write `/submission/submission.json`.
