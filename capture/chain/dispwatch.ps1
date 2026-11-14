# Diagnostic (2026-11-10): logs the display state Windows sends to applications (GUID_CONSOLE_DISPLAY_STATE,
# GUID_SESSION_DISPLAY_STATUS, GUID_MONITOR_POWER_ON) and session lock/unlock/connect events (WTS), with UTC times.
# These are what Chrome uses besides window overlap to mark its windows hidden. No keys, no text, no titles.
$logf = 'C:\work\v7-chain\dispwatch.log'
Add-Type -ReferencedAssemblies System.Windows.Forms @'
using System; using System.IO; using System.Runtime.InteropServices; using System.Windows.Forms;
public class DW : Form {
  [DllImport("user32.dll")] static extern IntPtr RegisterPowerSettingNotification(IntPtr h, ref Guid g, int f);
  [DllImport("wtsapi32.dll")] static extern bool WTSRegisterSessionNotification(IntPtr h, int f);
  [StructLayout(LayoutKind.Sequential, Pack = 4)] struct PBS { public Guid g; public uint len; public byte d0; }
  static Guid CONSOLE = new Guid("6FE69556-704A-47A0-8F24-C28D936FDA47");
  static Guid SESSION = new Guid("2B84C20E-AD23-4DDF-93DB-05FFBD7EFCA5");
  static Guid MONITOR = new Guid("02731015-4510-4526-99E6-E5A17EBD1AEA");
  string path;
  public DW(string p) { path = p; ShowInTaskbar = false; WindowState = FormWindowState.Minimized; Opacity = 0; }
  void Log(string m) { File.AppendAllText(path, DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ss") + "Z " + m + Environment.NewLine); }
  protected override void OnHandleCreated(EventArgs e) {
    base.OnHandleCreated(e);
    Guid a = CONSOLE, b = SESSION, c = MONITOR;
    RegisterPowerSettingNotification(Handle, ref a, 0); RegisterPowerSettingNotification(Handle, ref b, 0);
    RegisterPowerSettingNotification(Handle, ref c, 0); WTSRegisterSessionNotification(Handle, 0);
    Log("dispwatch start");
  }
  protected override void WndProc(ref Message m) {
    if (m.Msg == 0x218 && (int)m.WParam == 0x8013) {
      PBS s = (PBS)Marshal.PtrToStructure(m.LParam, typeof(PBS));
      string n = s.g == CONSOLE ? "console_display" : s.g == SESSION ? "session_display" : s.g == MONITOR ? "monitor_power" : s.g.ToString();
      Log(n + "=" + s.d0 + " (0 off, 1 on, 2 dimmed)");
    } else if (m.Msg == 0x2B1) {
      Log("WTS session change " + (int)m.WParam + " (5 logon, 6 logoff, 7 lock, 8 unlock, 1/2 console connect/disconnect, 3/4 remote connect/disconnect)");
    }
    base.WndProc(ref m);
  }
}
'@
[System.Windows.Forms.Application]::Run((New-Object DW $logf))
