use strict;
use warnings;

# Engine: pdflatex + biber (project uses biblatex)
$pdf_mode = 1;
$bibtex_use = 2;
$biber = 'biber %O %S';

# Keep build artifacts out of the source folder
$out_dir = 'build';
$aux_dir = 'build';

# Avoid stray file-lock issues on Windows
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 -file-line-error %O %S';

# Continuous preview mode (-pvc): open the PDF with SumatraPDF, which
# auto-reloads on rebuild and doesn't lock the file like Adobe Reader does.
$pdf_previewer = '"C:/Users/Work/AppData/Local/SumatraPDF/SumatraPDF.exe" -reuse-instance %O %S';

# Clean up extra aux extensions biblatex/biber generate
push @generated_exts, 'bbl', 'bcf', 'run.xml';

# After every successful compile, publish the PDF to docs/InversePhysics.pdf
# so the repo always ships the latest rendered notes without shipping build/.
# Copy to a temp file first and rename over the destination: a plain
# "copy /Y" can be interrupted mid-write if SumatraPDF (or AV/indexing)
# has the destination open, corrupting it; rename is atomic on the same volume.
$success_cmd = 'copy /Y "build\\main.pdf" "..\\InversePhysics.pdf.tmp" && move /Y "..\\InversePhysics.pdf.tmp" "..\\InversePhysics.pdf"';
