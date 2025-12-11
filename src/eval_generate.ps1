# ------------------------------------------------------------
# Script PowerShell: Ejecución Automática (Ubicación: src)
# ------------------------------------------------------------

# Obtener la ruta donde está guardado este script (.ps1)
$SCRIPT_DIR = $PSScriptRoot

# Nombres de archivos .py (Se asume que están en la misma carpeta que este script)
$FILE_GESTURE_GLOBAL = "eval_gesture.py"
$FILE_GESTURE_ISOLATED = "eval_isolated_gesture.py"
$FILE_PATHOLOGY = "eval_pathology_global.py"

# ----------------------------------
# 3. EVALUACIÓN DE PATOLOGÍA (GLOBAL)
# ----------------------------------
Write-Host "`n[3/3] Ejecutando: $FILE_PATHOLOGY" -ForegroundColor White

if (Test-Path $FILE_PATHOLOGY) {
    py $FILE_PATHOLOGY
    Run-GitCommit "Clasificacion de Patologias"
} else {
    Write-Error "No se encuentra el archivo: $FILE_PATHOLOGY"
}

# ----------------------------------
# FIN
# ----------------------------------
Write-Host "`n==============================================" -ForegroundColor Green
Write-Host " LISTO. QUE DESCANSES. " -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Green