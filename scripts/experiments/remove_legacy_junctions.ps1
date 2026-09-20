param(
    [string]$PathMap = "outputs/experiments/00_registry/path_map.csv"
)

$root = (Resolve-Path "outputs/experiments").Path
$rows = Import-Csv $PathMap | Where-Object { $_.status -eq "COMPATIBILITY_LINK" }
if ($rows.Count -ne 38) {
    throw "expected 38 compatibility junctions, got $($rows.Count)"
}

foreach ($row in $rows) {
    $old = [IO.Path]::GetFullPath($row.old_path)
    $new = [IO.Path]::GetFullPath($row.new_path)
    if (-not $old.StartsWith($root + "\", [StringComparison]::OrdinalIgnoreCase)) {
        throw "unsafe legacy path: $old"
    }
    if (-not $new.StartsWith($root + "\", [StringComparison]::OrdinalIgnoreCase)) {
        throw "unsafe organized target: $new"
    }
    $item = Get-Item -LiteralPath $old -Force -ErrorAction Stop
    if ($item.LinkType -ne "Junction") {
        throw "refusing to remove non-junction: $old"
    }
    if (-not (Test-Path -LiteralPath $new)) {
        throw "organized target missing: $new"
    }
}

$removed = 0
foreach ($row in $rows) {
    $old = [IO.Path]::GetFullPath($row.old_path)
    Remove-Item -LiteralPath $old -Force -ErrorAction Stop
    if (Test-Path -LiteralPath $old) {
        throw "junction still exists after removal: $old"
    }
    $removed++
}

# 将迁移状态写回 path_map，保留审计轨迹但不再声称兼容入口仍存在。
$mapRows = Import-Csv $PathMap
foreach ($row in $mapRows) {
    if ($row.status -eq "COMPATIBILITY_LINK") {
        $row.status = "LEGACY_PATH_REMOVED"
    }
}
$mapRows | Export-Csv $PathMap -NoTypeInformation -Encoding UTF8
Write-Output "removed_junctions=$removed"
