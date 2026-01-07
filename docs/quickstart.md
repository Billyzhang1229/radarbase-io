# Quickstart

This page will include basic installation and usage examples once the public
API is finalized.

## Storage configuration

Configure the data root with an `fsspec` URL so you can target SFTP, S3, or the
local filesystem. Keep credentials in native tooling (SSH config, AWS
credentials), and store only non-secret settings in `.env.local`.

Example `.env.local`:

```
RADARBASE_STORAGE_URL=sftp://radarbase/path/to/project
RADARBASE_STORAGE_OPTIONS={"anon": false}
```
