# Annotator setup

The code repository and synthetic demo can be shared publicly. The coordinator sends each annotator three private items separately:

1. that annotator's `study.sqlite` file;
2. authorized access to `clip_224x224_16f.zip` and its password;
3. the annotator code printed in `ANNOTATOR_README.txt`.

## First launch

Install Python 3.11 or 3.12. Put the private database at `local_data/study.sqlite`. Copy `config/annotator.example.toml` to `config/project.toml`, set the local archive path and password, and then use the launcher for the operating system:

- macOS: double-click `scripts/start_mac.command`;
- Windows: double-click `scripts/start_windows.bat`.

The browser address is local to the annotator's computer. Sign in with the supplied code. The first 56 items are the shared calibration set; the 28 single frames appear before the 28 sequences.

## Work sessions

- Finish calibration and wait for the coordinator's scale discussion before beginning the main queue.
- Complete the isolated-frame block before the continuous-sequence block.
- Use short sessions and take breaks. Closing the browser does not erase progress.
- Do not rename the database while the app is open.
- Do not discuss main-study clips with another annotator.

## Return results

Close the browser window and the launcher terminal, then run:

```bash
python manage.py export --annotator R01 --output exports/R01
```

Replace `R01` with the assigned code. Return the resulting export folder to the coordinator. The export contains ratings and task identifiers, not DFEW images.
