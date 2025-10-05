# Wild Agent Examples

This directory contains example files to help you get started with Wild Agent.

## Files

- **`samples.txt`**: Example reference text samples about AI/ML topics
- **`urls.txt`**: Example URLs to crawl for similar content
- **`expected-output.json`**: Example of the JSON export format

## Usage

### Basic Example

Use the provided samples to find similar content:

```bash
wild-agent collect --sample-file examples/samples.txt
```

### With URLs

Crawl the example URLs and rank by similarity:

```bash
wild-agent collect \
  --sample-file examples/samples.txt \
  --urls-file examples/urls.txt
```

### Export Results

Save results to JSON:

```bash
wild-agent collect \
  --sample-file examples/samples.txt \
  --urls-file examples/urls.txt \
  --export my-results.json
```

### Adjust Parameters

Control the number of results and similarity threshold:

```bash
wild-agent collect \
  --sample-file examples/samples.txt \
  --top-n 5 \
  --threshold 0.8
```

### Only Crawl (No Online Search)

Disable online search to only crawl provided URLs:

```bash
wild-agent collect \
  --sample-file examples/samples.txt \
  --urls-file examples/urls.txt \
  --no-online-search
```

### Only Search (No Crawling)

Disable crawling to only use online search:

```bash
wild-agent collect \
  --sample-file examples/samples.txt \
  --no-crawl
```

### Verbose Output

Enable detailed logging:

```bash
wild-agent collect \
  --sample-file examples/samples.txt \
  --verbose
```

## File Formats

### samples.txt

```
# Comments start with #
Your first sample text here...
Your second sample text here...
```

### urls.txt

```
# Comments start with #
https://example.com/page1
https://example.com/page2
```

## Expected Output

The CLI displays:

1. **Analysis Phase**: Theme extraction from your samples
2. **Collection Phase**: Progress of online search and crawling
3. **Ranking Phase**: Similarity calculation and filtering
4. **Results**: Top-N ranked samples with scores
5. **Summary**: Statistics about the entire process

### JSON Export Format

When using `--export`, results are saved in JSON format with:

- **analysis**: Theme extraction details
- **collection**: Collection statistics and metadata
- **ranking**: Ranked results with scores and sample details

See `expected-output.json` for a complete example.

## Tips

1. **API Key**: Make sure to set `XAI_API_KEY` environment variable
2. **Samples**: Provide 1-10 reference samples for best results
3. **Threshold**: Start with 0.7, adjust based on results (higher = more strict)
4. **Top-N**: Default is 10, increase if you need more results
5. **URLs**: Crawling works best with content-rich pages (articles, documentation)

## Troubleshooting

### No results returned

- Lower the `--threshold` value
- Increase `--top-n` value
- Check that your samples are substantial (50+ characters)

### Rate limit errors

- The tool handles rate limits automatically with retries
- Wait a few minutes between large requests

### Crawling failures

- Some URLs may block crawlers
- Check the summary for failed URL count
- Use `--verbose` to see detailed error messages

## More Information

- **Main Documentation**: `../README.md`
- **Quickstart Guide**: `../specs/001-sample-collection-and/quickstart.md`
- **CLI Help**: `wild-agent collect --help`
