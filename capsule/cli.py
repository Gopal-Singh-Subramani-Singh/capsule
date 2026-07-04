from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
import structlog

app = typer.Typer(
    name="capsule",
    help="Container-native ML model deployment platform",
    add_completion=False,
)
console = Console()
logger = structlog.get_logger(__name__)


def _get_components(config_path: str = "config/defaults.yaml"):
    """Initialise all Capsule components from config."""
    import yaml
    from capsule.registry import ModelRegistry
    from capsule.manifest import ManifestStore
    from capsule.k8s_client import K8sClient
    from capsule.helm import HelmChartGenerator
    from capsule.deployer import Deployer
    from capsule.packager import Packager

    cfg = {}
    if Path(config_path).exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}

    reg_cfg = cfg.get("registry", {})
    k8s_cfg = cfg.get("k8s", {})
    docker_cfg = cfg.get("docker", {})
    helm_cfg = cfg.get("helm", {})
    onnx_cfg = cfg.get("onnx", {})

    registry = ModelRegistry(
        endpoint=reg_cfg.get("endpoint", "http://localhost:9000"),
        access_key=reg_cfg.get("access_key", "minioadmin"),
        secret_key=reg_cfg.get("secret_key", "minioadmin"),
        bucket=reg_cfg.get("bucket", "capsule-models"),
    )
    store = ManifestStore()
    k8s = K8sClient(namespace=k8s_cfg.get("namespace", "capsule"))
    helm_gen = HelmChartGenerator(
        chart_dir=helm_cfg.get("chart_dir", "/tmp/capsule-charts")
    )
    deployer = Deployer(
        registry=registry, store=store, k8s=k8s,
        helm_gen=helm_gen, namespace=k8s_cfg.get("namespace", "capsule"),
    )
    packager = Packager(
        registry=registry, store=store,
        docker_registry=docker_cfg.get("registry", "localhost:5001"),
        onnx_enabled=onnx_cfg.get("enabled", True),
    )
    return registry, store, k8s, deployer, packager


@app.command("package")
def cmd_package(
    manifest_path: str = typer.Option("capsule.yaml", "--manifest", "-m"),
    no_onnx: bool = typer.Option(False, "--no-onnx", help="Skip ONNX optimisation"),
    no_push: bool = typer.Option(False, "--no-push", help="Build only, do not push"),
):
    """Package a model: detect framework, build Docker image, optimise, push to registry."""
    from capsule.manifest import load_manifest

    console.print("\n[bold cyan]Capsule Package[/bold cyan]\n")

    try:
        manifest = load_manifest(manifest_path)
    except FileNotFoundError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1)

    console.print(f"  Model:     [bold]{manifest.name}:{manifest.version}[/bold]")
    console.print(f"  Model path: {manifest.model_path}")

    _, store, _, _, packager = _get_components()
    packager._onnx_enabled = not no_onnx

    try:
        with console.status("Building..."):
            result = packager.package(
                manifest,
                push_image=not no_push,
                manifest_dir=str(Path(manifest_path).parent.resolve()),
            )
    except Exception as exc:
        console.print(f"\n[red]✗ Package failed:[/red] {exc}")
        raise typer.Exit(1)

    size_str = (
        f"{result.original_size_mb * 1024:.0f} KB"
        if result.original_size_mb < 1.0
        else f"{result.original_size_mb:.1f} MB"
    )
    console.print(f"\n[green]✓[/green] Packaged [bold]{result.name}:{result.version}[/bold]")
    console.print(f"  Image:    {result.image_tag}")
    console.print(f"  Framework: {result.framework.value}")
    console.print(
        f"  Size:     {size_str}"
        + (
            f" → {result.optimised_size_mb:.1f} MB "
            f"([green]-{result.size_reduction_pct:.0f}%[/green])"
            if result.onnx_optimised and result.size_reduction_pct is not None else (
                f" → {result.optimised_size_mb:.1f} MB" if result.onnx_optimised else ""
            )
        )
    )
    console.print(f"  Build time: {result.build_seconds:.1f}s")


@app.command("deploy")
def cmd_deploy(
    name_version: str = typer.Argument(
        ..., help="name:version (e.g. fraud-detector:2.1)"
    ),
    canary: int = typer.Option(0, "--canary", "-c", help="Canary traffic % (0=full)"),
    manifest_path: str = typer.Option("capsule.yaml", "--manifest", "-m"),
    namespace: str = typer.Option("capsule", "--namespace", "-n"),
):
    """Deploy a packaged model to K3s with optional canary splitting."""
    from capsule.manifest import load_manifest

    console.print("\n[bold cyan]Capsule Deploy[/bold cyan]\n")

    if ":" not in name_version:
        console.print("[red]✗ Format must be name:version[/red]")
        raise typer.Exit(1)

    name, version = name_version.rsplit(":", 1)
    registry, store, _, deployer, _ = _get_components()

    # Get image tag from registry
    image_tag = registry.get_image_tag(name, version)
    if not image_tag:
        console.print(f"[red]✗ No image found for {name}:{version}[/red]")
        console.print("  Run: capsule package first")
        raise typer.Exit(1)

    try:
        manifest = load_manifest(manifest_path)
    except FileNotFoundError:
        from capsule.models import CapsuleManifest
        manifest = CapsuleManifest(
            name=name, version=version, model_path="model.pt"
        )

    console.print(f"  Deploying: [bold]{name}:{version}[/bold]")
    console.print(f"  Image:     {image_tag}")
    if canary > 0:
        console.print(f"  Canary:    [yellow]{canary}%[/yellow] traffic to new version")

    try:
        with console.status("Deploying to K3s..."):
            record = deployer.deploy(manifest, image_tag, canary_weight=canary)
    except Exception as exc:
        console.print(f"\n[red]✗ Deploy failed:[/red] {exc}")
        raise typer.Exit(1)
    console.print(f"\n[green]✓[/green] Deployed [bold]{name}:{version}[/bold]")
    console.print(f"  Status:    {record.status.value}")
    if canary > 0:
        console.print(
            f"  Canary:    {canary}% — use [bold]capsule status {name}[/bold] to monitor"
        )


@app.command("status")
def cmd_status(
    name: str = typer.Argument(..., help="Deployment name"),
):
    """Show detailed deployment status."""
    console.print(f"\n[bold cyan]Capsule Status — {name}[/bold cyan]\n")

    _, store, _, deployer, _ = _get_components()

    try:
        status = deployer.get_status(name)
    except Exception as exc:
        console.print(f"[red]✗ Could not get status:[/red] {exc}")
        raise typer.Exit(1)

    # Summary panel
    status_color = {
        "running": "green", "canary": "yellow",
        "failed": "red", "rolled_back": "orange3", "pending": "blue",
    }.get(status.status.value, "white")

    console.print(Panel(
        f"[{status_color}]{status.status.value.upper()}[/{status_color}]  "
        f"stable=[bold]{status.stable_version or 'none'}[/bold]  "
        f"canary=[bold]{status.canary_version or 'none'}[/bold]  "
        f"canary_traffic=[bold]{status.canary_weight}%[/bold]",
        title=name, border_style=status_color,
    ))

    # Pods table
    if status.pods:
        t = Table(title="Pods", box=box.SIMPLE)
        t.add_column("Name", style="cyan")
        t.add_column("Phase")
        t.add_column("Ready")
        t.add_column("Restarts", justify="right")
        t.add_column("Age")
        t.add_column("Version")
        for pod in status.pods:
            age = f"{int(pod.age_seconds // 60)}m"
            t.add_row(
                pod.name[:40],
                pod.phase,
                "[green]✓[/green]" if pod.ready else "[red]✗[/red]",
                str(pod.restarts),
                age,
                pod.version,
            )
        console.print(t)

    # Events
    if status.events:
        console.print("\n[bold]Recent events:[/bold]")
        for event in status.events[:5]:
            console.print(f"  [dim]•[/dim] {event}")


@app.command("rollback")
def cmd_rollback(
    name: str = typer.Argument(..., help="Deployment name to roll back"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Roll back a deployment to the previous stable version."""
    console.print(f"\n[bold cyan]Capsule Rollback — {name}[/bold cyan]\n")

    _, store, _, deployer, _ = _get_components()

    current = store.get_latest_deployment(name)
    if not current:
        console.print(f"[red]✗ No deployment found for {name}[/red]")
        raise typer.Exit(1)

    prev = store.get_previous_version(name, current.version)
    if not prev:
        console.print(f"[red]✗ No previous version found for {name}[/red]")
        raise typer.Exit(1)

    console.print(f"  Current version: [yellow]{current.version}[/yellow]")
    console.print(f"  Rolling back to: [green]{prev}[/green]")

    if not confirm:
        confirmed = typer.confirm("\nProceed with rollback?")
        if not confirmed:
            console.print("[yellow]Rollback cancelled[/yellow]")
            raise typer.Exit(0)

    try:
        with console.status("Rolling back..."):
            result = deployer.rollback(name)
    except Exception as exc:
        console.print(f"\n[red]✗ Rollback failed:[/red] {exc}")
        raise typer.Exit(1)

    console.print(
        f"\n[green]✓[/green] Rolled back [bold]{name}[/bold] "
        f"from {result['rolled_back_from']} → {result['rolled_back_to']} "
        f"in {result['duration_seconds']:.1f}s"
    )


@app.command("list")
def cmd_list(
    name: Optional[str] = typer.Argument(None, help="Filter by model name"),
    limit: int = typer.Option(50, "--limit", "-l", help="Max rows to show"),
):
    """List packaged models in the registry."""
    _, store, _, _, _ = _get_components()
    packages = store.list_packages(name=name, limit=limit)

    if not packages:
        console.print("[dim]No packages found[/dim]")
        return

    t = Table(title="Packaged Models", box=box.SIMPLE)
    t.add_column("Name", style="cyan")
    t.add_column("Version")
    t.add_column("Framework")
    t.add_column("Size (MB)", justify="right")
    t.add_column("ONNX", justify="center")
    t.add_column("Digest", style="dim")
    t.add_column("Build (s)", justify="right")
    t.add_column("Packaged At")
    for p in packages:
        digest_short = (p.get("image_digest") or "—")
        if digest_short and digest_short.startswith("sha256:"):
            digest_short = digest_short[7:19] + "…"
        t.add_row(
            p["name"], p["version"], p["framework"],
            f"{p['original_size_mb']:.1f}",
            "[green]✓[/green]" if p["onnx_optimised"] else "—",
            digest_short,
            f"{p['build_seconds']:.0f}",
            str(p["packaged_at"])[:16],
        )
    console.print(t)


@app.command("watch")
def cmd_watch(
    name: str = typer.Argument(..., help="Deployment name to watch"),
    interval: int = typer.Option(30, "--interval", "-i", help="Check interval seconds"),
    threshold: float = typer.Option(0.05, "--threshold", "-t", help="Error rate rollback threshold"),
    windows: int = typer.Option(10, "--windows", "-w", help="Healthy windows before auto-promote"),
    failures: int = typer.Option(2, "--failures", "-f", help="Consecutive failures before rollback"),
    prometheus: str = typer.Option("http://localhost:9090", "--prometheus", "-p"),
):
    """Watch a canary deployment and auto-rollback or promote based on error rate."""
    import asyncio

    registry, store, _, deployer, _ = _get_components()
    from capsule.canary import CanaryController

    record = store.get_latest_deployment(name)
    if not record:
        console.print(f"[red]✗ No deployment found for {name}[/red]")
        raise typer.Exit(1)

    if record.canary_weight == 0:
        console.print(f"[yellow]⚠[/yellow]  {name} is not in canary mode (weight=0)")
        console.print("  Deploy with --canary N first, then run capsule watch.")
        raise typer.Exit(1)

    canary_url = f"http://capsule-{name}:8080"
    console.print(f"\n[bold cyan]Capsule Watch — {name}[/bold cyan]")
    console.print(f"  Canary weight:   {record.canary_weight}%")
    console.print(f"  Error threshold: {threshold * 100:.1f}%")
    console.print(f"  Promote after:   {windows} clean windows")
    console.print(f"  Rollback after:  {failures} bad windows")
    console.print(f"  Check interval:  {interval}s\n")
    console.print("[dim]Ctrl+C to stop watching[/dim]\n")

    def do_rollback():
        console.print(f"\n[red]⚡ Auto-rollback triggered for {name}[/red]")
        try:
            result = deployer.rollback(name)
            console.print(
                f"[green]✓[/green] Rolled back {name}: "
                f"{result['rolled_back_from']} → {result['rolled_back_to']}"
            )
        except Exception as exc:
            console.print(f"[red]✗ Rollback failed:[/red] {exc}")

    def do_promote():
        console.print(f"\n[green]✓ Auto-promote triggered for {name}[/green]")
        console.print("  Re-deploy at 100% traffic: capsule deploy "
                      f"{name}:{record.version}")

    ctrl = CanaryController(
        store=store,
        k8s=_get_components()[2],
        deployment_name=name,
        canary_service_url=canary_url,
        error_rate_threshold=threshold,
        monitor_interval_seconds=interval,
        consecutive_failures=failures,
        auto_promote_windows=windows,
        prometheus_url=prometheus,
        auto_rollback_fn=do_rollback,
        auto_promote_fn=do_promote,
    )

    async def run():
        await ctrl.start()
        try:
            while ctrl._running:
                await asyncio.sleep(1)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            await ctrl.stop()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        console.print("\n[dim]Watch stopped[/dim]")


@app.command("audit")
def cmd_audit(
    limit: int = typer.Option(30, "--limit", "-l", help="Number of entries to show"),
):
    """Show the audit log of all package and deploy operations."""
    _, store, _, _, _ = _get_components()
    entries = store.get_audit_log(limit=limit)

    if not entries:
        console.print("[dim]No audit entries[/dim]")
        return

    t = Table(title="Audit Log", box=box.SIMPLE)
    t.add_column("Time", style="dim")
    t.add_column("Action", style="bold")
    t.add_column("Target", style="cyan")
    t.add_column("Detail")
    t.add_column("Actor")
    for e in entries:
        t.add_row(
            str(e.get("ts", ""))[:19],
            e.get("action", ""),
            e.get("target", ""),
            e.get("detail", ""),
            e.get("actor", "cli"),
        )
    console.print(t)


if __name__ == "__main__":
    app()
