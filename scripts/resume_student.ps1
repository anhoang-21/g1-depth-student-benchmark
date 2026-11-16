# Continues the G1 depth-student distillation from its newest checkpoint (student + teacher + optimizer + iteration).
# Run in a normal PowerShell window:
#   powershell -ExecutionPolicy Bypass -File C:\Users\ADMIN\Isaacsim\resume_student.ps1
# Stop with Ctrl+C (a checkpoint is saved every 250 iterations); run the same command again to continue.
# The training processes are opted out of Windows background throttling and set to High priority, so they
# keep using the performance cores when another window has the focus.
param([int]$Envs = 2048, [int]$Total = 24000)
$lab = "C:\Users\ADMIN\Isaacsim\IsaacLab"
$env:OMNI_KIT_ACCEPT_EULA = "YES"
$env:PATH = "C:\Program Files\Git\cmd;" + $env:PATH
Set-Location $lab

$ckpt = Get-ChildItem "$lab\logs\rsl_rl\g1_rough_depth_student\*\model_*.pt" |
    Sort-Object @{ e = { [int]($_.BaseName -replace "model_", "") } }, LastWriteTime | Select-Object -Last 1
if (-not $ckpt) { Write-Host "No student checkpoint found."; exit 1 }
$done = [int]($ckpt.BaseName -replace "model_", "")
$left = $Total - $done
if ($left -le 0) { Write-Host "Student already reached $done / $Total iterations: $($ckpt.FullName)"; exit 0 }

Add-Type -Namespace TrainQos -Name Native -MemberDefinition @'
[StructLayout(LayoutKind.Sequential)] public struct PT { public uint Version; public uint ControlMask; public uint StateMask; }
[DllImport("kernel32.dll", SetLastError=true)] public static extern IntPtr OpenProcess(uint access, bool inherit, int pid);
[DllImport("kernel32.dll", SetLastError=true)] public static extern bool SetProcessInformation(IntPtr h, int cls, ref PT info, int size);
[DllImport("kernel32.dll")] public static extern bool CloseHandle(IntPtr h);
public static bool HighQos(int pid) {
    IntPtr h = OpenProcess(0x0200, false, pid);
    if (h == IntPtr.Zero) return false;
    PT p = new PT(); p.Version = 1; p.ControlMask = 1; p.StateMask = 0;   // execution-speed throttling explicitly off
    bool ok = SetProcessInformation(h, 4, ref p, 12);
    CloseHandle(h);
    return ok;
}
'@

Write-Host "Resuming from $($ckpt.FullName)"
Write-Host "Iterations done: $done, remaining: $left, envs: $Envs"
$argv = @("train", "--rl_library", "rsl_rl", "--task", "Isaac-Velocity-Rough-G1-Depth-Student",
          "--checkpoint", $ckpt.FullName, "--num_envs", "$Envs", "--max_iterations", "$left")
$p = Start-Process -FilePath "$lab\env_isaaclab\Scripts\isaaclab.exe" -ArgumentList $argv -WorkingDirectory $lab -NoNewWindow -PassThru

# Boost the launcher and the Python processes it spawns (they appear within the first minute).
$boosted = @{}
$until = (Get-Date).AddMinutes(3)
while (-not $p.HasExited -and (Get-Date) -lt $until) {
    Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "Depth-Student" -and $_.Name -in "python.exe", "isaaclab.exe" } | ForEach-Object {
        if (-not $boosted.ContainsKey($_.ProcessId)) {
            $ok = [TrainQos.Native]::HighQos($_.ProcessId)
            try { (Get-Process -Id $_.ProcessId).PriorityClass = "High" } catch {}
            $boosted[$_.ProcessId] = $ok
        }
    }
    Start-Sleep -Seconds 5
}
$p.WaitForExit()
