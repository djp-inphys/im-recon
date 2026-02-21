Get-ChildItem -File -Filter *.png | Sort-Object {
    if ($_.Name -match 'instance_(\d+)') {
        [int]$matches[1]
    }
    else {
        0
    }
} |ForEach-Object {
    "file '$($_.FullName)'"
} | Set-Content -Path frames.txt -Encoding ascii