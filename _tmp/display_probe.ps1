# Enumerate display devices and per-device available modes (why the resolution list is locked).
# ASCII-only on purpose: Windows PowerShell 5.1 reads UTF-8-without-BOM as GBK and breaks on non-ASCII.
Add-Type -AssemblyName System.Windows.Forms

$sig = @'
using System;
using System.Runtime.InteropServices;
using System.Text;

public class Disp {
  [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
  public struct DISPLAY_DEVICE {
    public int cb;
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)] public string DeviceName;
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)] public string DeviceString;
    public int StateFlags;
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)] public string DeviceID;
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)] public string DeviceKey;
  }

  [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
  public struct DEVMODE {
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)] public string dmDeviceName;
    public short dmSpecVersion, dmDriverVersion, dmSize, dmDriverExtra;
    public int dmFields;
    public int dmPositionX, dmPositionY;
    public int dmDisplayOrientation, dmDisplayFixedOutput;
    public short dmColor, dmDuplex, dmYResolution, dmTTOption, dmCollate;
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)] public string dmFormName;
    public short dmLogPixels;
    public int dmBitsPerPel, dmPelsWidth, dmPelsHeight, dmDisplayFlags, dmDisplayFrequency;
    public int dmICMMethod, dmICMIntent, dmMediaType, dmDitherType, dmReserved1, dmReserved2, dmPanningWidth, dmPanningHeight;
  }

  [DllImport("user32.dll", CharSet = CharSet.Unicode)]
  public static extern bool EnumDisplayDevices(string lpDevice, uint iDevNum, ref DISPLAY_DEVICE lpDisplayDevice, uint dwFlags);

  [DllImport("user32.dll", CharSet = CharSet.Unicode)]
  public static extern bool EnumDisplaySettings(string lpszDeviceName, int iModeNum, ref DEVMODE lpDevMode);
}
'@
Add-Type -TypeDefinition $sig

$ATTACHED = 0x1
$PRIMARY = 0x4
$MIRROR = 0x8
$CURRENT = -1

Write-Host "===== A. WinForms active screens ====="
[System.Windows.Forms.Screen]::AllScreens | ForEach-Object {
  "  {0}  Primary={1}  Bounds={2}x{3} @ ({4},{5})" -f $_.DeviceName, $_.Primary, $_.Bounds.Width, $_.Bounds.Height, $_.Bounds.X, $_.Bounds.Y
}

Write-Host ""
Write-Host "===== B. adapters / monitors / modes ====="
$i = 0
while ($true) {
  $d = New-Object Disp+DISPLAY_DEVICE
  $d.cb = [System.Runtime.InteropServices.Marshal]::SizeOf($d)
  if (-not [Disp]::EnumDisplayDevices($null, $i, [ref]$d, 0)) { break }
  $i++

  $flags = @()
  if ($d.StateFlags -band $ATTACHED) { $flags += "ATTACHED" }
  if ($d.StateFlags -band $PRIMARY) { $flags += "PRIMARY" }
  if ($d.StateFlags -band $MIRROR) { $flags += "MIRROR" }
  $flagText = if ($flags.Count) { $flags -join "+" } else { "detached/not-active" }

  Write-Host ("")
  Write-Host ("  [{0}] {1}" -f $d.DeviceName, $flagText)
  Write-Host ("      driver : {0}" -f $d.DeviceString)
  Write-Host ("      device : {0}" -f $d.DeviceID)

  $j = 0
  while ($true) {
    $m = New-Object Disp+DISPLAY_DEVICE
    $m.cb = [System.Runtime.InteropServices.Marshal]::SizeOf($m)
    if (-not [Disp]::EnumDisplayDevices($d.DeviceName, $j, [ref]$m, 0)) { break }
    Write-Host ("      monitor[{0}]: {1} | {2} | flags=0x{3:X}" -f $j, $m.DeviceString, $m.DeviceID, $m.StateFlags)
    $j++
  }

  $cur = New-Object Disp+DEVMODE
  $cur.dmSize = [short][System.Runtime.InteropServices.Marshal]::SizeOf($cur)
  if ([Disp]::EnumDisplaySettings($d.DeviceName, $CURRENT, [ref]$cur)) {
    Write-Host ("      current: {0}x{1} @{2}Hz {3}bpp pos=({4},{5})" -f $cur.dmPelsWidth, $cur.dmPelsHeight, $cur.dmDisplayFrequency, $cur.dmBitsPerPel, $cur.dmPositionX, $cur.dmPositionY)
  }

  $modes = @{}
  $k = 0
  while ($true) {
    $dm = New-Object Disp+DEVMODE
    $dm.dmSize = [short][System.Runtime.InteropServices.Marshal]::SizeOf($dm)
    if (-not [Disp]::EnumDisplaySettings($d.DeviceName, $k, [ref]$dm)) { break }
    $key = "{0}x{1}" -f $dm.dmPelsWidth, $dm.dmPelsHeight
    if (-not $modes.ContainsKey($key)) { $modes[$key] = 0 }
    $modes[$key]++
    $k++
  }
  Write-Host ("      modes: {0} distinct resolutions / {1} total" -f $modes.Count, $k)
  ($modes.Keys | Sort-Object { [int]($_ -split 'x')[0] }) | ForEach-Object { Write-Host ("         - {0}" -f $_) }
}
