"""
3D Mesh Generation & Reconstruction for RealSense D455i (CPU-Only)

This module provides comprehensive mesh generation, processing, and export
capabilities using Open3D, trimesh, and scikit-image.

Author: PGG Desktop App Team
Date: 2025-10-10
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Literal
from dataclasses import dataclass

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("⚠️  Open3D not available. Mesh generation features limited.")

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("⚠️  trimesh not available. Some mesh features disabled.")

try:
    from skimage import measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("⚠️  scikit-image not available. Marching cubes disabled.")

try:
    from scipy.spatial import Delaunay
    from scipy.spatial import ConvexHull
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️  scipy not available. Delaunay triangulation disabled.")


@dataclass
class MeshQualityMetrics:
    """Quality metrics for mesh analysis."""
    triangle_count: int
    vertex_count: int
    edge_count: int
    is_watertight: bool
    is_manifold: bool
    mean_triangle_aspect_ratio: float
    min_triangle_area: float
    max_triangle_area: float
    surface_area: float
    volume: float


class MeshGenerator:
    """
    Advanced mesh generation and processing for RealSense point clouds (CPU-only).

    Features:
    - Surface reconstruction (Poisson, Ball Pivoting, Marching Cubes)
    - Triangulation (Delaunay, Alpha Shapes)
    - Mesh simplification and smoothing
    - Hole filling and topology repair
    - Quality analysis and validation
    - Multi-format export (OBJ, STL, PLY, COLLADA, OFF)
    """

    def __init__(self):
        """Initialize mesh generator."""
        self.current_mesh = None

    # ========================================================================
    # SURFACE RECONSTRUCTION
    # ========================================================================

    def apply_poisson_reconstruction(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        depth: int = 9,
        width: int = 0,
        scale: float = 1.1,
        linear_fit: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Poisson surface reconstruction.

        Creates a watertight mesh from oriented point cloud.

        Args:
            points: Point cloud [N, 3]
            normals: Normals [N, 3]
            depth: Octree depth (controls detail, 8-12 typical)
            width: Target width of finest level octree cells
            scale: Surface reconstruction scale
            linear_fit: Whether to use linear fit

        Returns:
            Tuple of (vertices [M, 3], triangles [T, 3])
        """
        if not OPEN3D_AVAILABLE:
            print("⚠️  Open3D required for Poisson reconstruction.")
            return np.array([]), np.array([])

        # Create Open3D point cloud with normals
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)

        # Apply Poisson reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=depth,
            width=width,
            scale=scale,
            linear_fit=linear_fit
        )

        # Store mesh
        self.current_mesh = mesh

        # Extract vertices and triangles
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        print(f"✅ Poisson reconstruction: {len(vertices)} vertices, {len(triangles)} triangles")

        return vertices, triangles

    def apply_marching_cubes(
        self,
        volume: np.ndarray,
        level: float = 0.0,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply marching cubes algorithm to extract isosurface.

        Args:
            volume: 3D volumetric data [X, Y, Z]
            level: Isosurface level (contour value)
            spacing: Voxel spacing in (x, y, z)

        Returns:
            Tuple of (vertices [M, 3], triangles [T, 3])
        """
        if not SKIMAGE_AVAILABLE:
            print("⚠️  scikit-image required for marching cubes.")
            return np.array([]), np.array([])

        # Apply marching cubes
        verts, faces, normals, values = measure.marching_cubes(
            volume,
            level=level,
            spacing=spacing
        )

        print(f"✅ Marching cubes: {len(verts)} vertices, {len(faces)} triangles")

        return verts, faces

    def apply_ball_pivoting(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        radii: List[float] = [0.005, 0.01, 0.02, 0.04]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply ball pivoting algorithm.

        Creates mesh that closely follows point cloud surface.

        Args:
            points: Point cloud [N, 3]
            normals: Normals [N, 3]
            radii: Ball radii to try (list of increasing radii)

        Returns:
            Tuple of (vertices [M, 3], triangles [T, 3])
        """
        if not OPEN3D_AVAILABLE:
            print("⚠️  Open3D required for ball pivoting.")
            return np.array([]), np.array([])

        # Create Open3D point cloud with normals
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)

        # Apply ball pivoting
        radii_vector = o3d.utility.DoubleVector(radii)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd,
            radii_vector
        )

        # Store mesh
        self.current_mesh = mesh

        # Extract vertices and triangles
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        print(f"✅ Ball pivoting: {len(vertices)} vertices, {len(triangles)} triangles")

        return vertices, triangles

    def apply_delaunay_triangulation(
        self,
        points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Delaunay triangulation.

        Creates optimal triangulation with no overlapping triangles.

        Args:
            points: Point cloud [N, 3] (or [N, 2] for 2D)

        Returns:
            Tuple of (vertices [M, 3], simplices [T, 3 or 4])
        """
        if not SCIPY_AVAILABLE:
            print("⚠️  scipy required for Delaunay triangulation.")
            return np.array([]), np.array([])

        # Apply Delaunay triangulation
        delaunay = Delaunay(points)

        print(f"✅ Delaunay triangulation: {len(points)} points, {len(delaunay.simplices)} simplices")

        return points, delaunay.simplices

    def generate_alpha_shape(
        self,
        points: np.ndarray,
        alpha: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate alpha shape (concave hull).

        Better than convex hull for complex shapes.

        Args:
            points: Point cloud [N, 3]
            alpha: Alpha parameter (controls tightness, smaller = tighter)

        Returns:
            Tuple of (vertices [M, 3], triangles [T, 3])
        """
        if not OPEN3D_AVAILABLE:
            print("⚠️  Open3D required for alpha shapes.")
            return np.array([]), np.array([])

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Compute alpha shape
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd,
            alpha
        )

        # Store mesh
        self.current_mesh = mesh

        # Extract vertices and triangles
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        print(f"✅ Alpha shape (alpha={alpha}): {len(vertices)} vertices, {len(triangles)} triangles")

        return vertices, triangles

    # ========================================================================
    # MESH PROCESSING
    # ========================================================================

    def simplify_mesh_by_decimation(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        target_reduction: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simplify mesh by quadric decimation.

        Args:
            vertices: Mesh vertices [N, 3]
            triangles: Mesh triangles [T, 3]
            target_reduction: Target reduction ratio (0.5 = 50% fewer triangles)

        Returns:
            Tuple of (simplified_vertices, simplified_triangles)
        """
        if not OPEN3D_AVAILABLE:
            print("⚠️  Open3D required for mesh decimation.")
            return vertices, triangles

        # Create Open3D mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)

        # Calculate target triangle count
        target_count = int(len(triangles) * (1 - target_reduction))

        # Apply quadric decimation
        mesh_simplified = mesh.simplify_quadric_decimation(
            target_number_of_triangles=target_count
        )

        # Extract simplified mesh
        simplified_vertices = np.asarray(mesh_simplified.vertices)
        simplified_triangles = np.asarray(mesh_simplified.triangles)

        print(f"✅ Decimation: {len(triangles)} → {len(simplified_triangles)} triangles "
              f"({target_reduction*100:.0f}% reduction)")

        return simplified_vertices, simplified_triangles

    def apply_laplacian_smoothing(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        iterations: int = 5
    ) -> np.ndarray:
        """
        Apply Laplacian smoothing to mesh.

        Reduces high-frequency noise but may cause shrinkage.

        Args:
            vertices: Mesh vertices [N, 3]
            triangles: Mesh triangles [T, 3]
            iterations: Number of smoothing iterations

        Returns:
            Smoothed vertices [N, 3]
        """
        if not OPEN3D_AVAILABLE:
            print("⚠️  Open3D required for Laplacian smoothing.")
            return vertices

        # Create Open3D mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)

        # Apply Laplacian smoothing
        mesh_smoothed = mesh.filter_smooth_laplacian(
            number_of_iterations=iterations
        )

        # Extract smoothed vertices
        smoothed_vertices = np.asarray(mesh_smoothed.vertices)

        print(f"✅ Laplacian smoothing: {iterations} iterations")

        return smoothed_vertices

    def apply_taubin_smoothing(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        iterations: int = 10,
        lamb: float = 0.5,
        mu: float = -0.53
    ) -> np.ndarray:
        """
        Apply Taubin smoothing to mesh.

        Smooths without shrinkage, better volume preservation than Laplacian.

        Args:
            vertices: Mesh vertices [N, 3]
            triangles: Mesh triangles [T, 3]
            iterations: Number of smoothing iterations
            lamb: Lambda parameter (forward step)
            mu: Mu parameter (backward step, should be negative)

        Returns:
            Smoothed vertices [N, 3]
        """
        if not TRIMESH_AVAILABLE:
            print("⚠️  trimesh required for Taubin smoothing.")
            return vertices

        # Create trimesh mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

        # Apply Taubin smoothing
        trimesh.smoothing.filter_taubin(mesh, iterations=iterations, lamb=lamb, mu=mu)

        # Extract smoothed vertices
        smoothed_vertices = mesh.vertices

        print(f"✅ Taubin smoothing: {iterations} iterations (λ={lamb}, μ={mu})")

        return smoothed_vertices

    def fill_holes_in_mesh(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        hole_size: float = 10.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fill holes in mesh.

        Detects holes and fills them with new triangles.

        Args:
            vertices: Mesh vertices [N, 3]
            triangles: Mesh triangles [T, 3]
            hole_size: Maximum hole size to fill

        Returns:
            Tuple of (filled_vertices, filled_triangles)
        """
        if not TRIMESH_AVAILABLE:
            print("⚠️  trimesh required for hole filling.")
            return vertices, triangles

        # Create trimesh mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

        # Fill holes
        trimesh.repair.fill_holes(mesh)

        # Extract filled mesh
        filled_vertices = mesh.vertices
        filled_triangles = mesh.faces

        print(f"✅ Hole filling: {len(triangles)} → {len(filled_triangles)} triangles")

        return filled_vertices, filled_triangles

    def repair_mesh_topology(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Repair mesh topology.

        Fixes non-manifold edges and removes degenerate triangles.

        Args:
            vertices: Mesh vertices [N, 3]
            triangles: Mesh triangles [T, 3]

        Returns:
            Tuple of (repaired_vertices, repaired_triangles)
        """
        if not TRIMESH_AVAILABLE:
            print("⚠️  trimesh required for mesh repair.")
            return vertices, triangles

        # Create trimesh mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

        # Repair mesh
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fix_inversion(mesh)
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
        mesh.remove_infinite_values()
        mesh.remove_unreferenced_vertices()

        # Extract repaired mesh
        repaired_vertices = mesh.vertices
        repaired_triangles = mesh.faces

        print(f"✅ Mesh repair: {len(vertices)} → {len(repaired_vertices)} vertices, "
              f"{len(triangles)} → {len(repaired_triangles)} triangles")

        return repaired_vertices, repaired_triangles

    # ========================================================================
    # MESH ANALYSIS
    # ========================================================================

    def compute_mesh_normals(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        per_vertex: bool = True
    ) -> np.ndarray:
        """
        Compute mesh normals.

        Args:
            vertices: Mesh vertices [N, 3]
            triangles: Mesh triangles [T, 3]
            per_vertex: If True, compute per-vertex normals; else per-face

        Returns:
            Normals [N, 3] or [T, 3] (unit vectors)
        """
        if not OPEN3D_AVAILABLE:
            print("⚠️  Open3D required for normal computation.")
            return np.array([])

        # Create Open3D mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)

        if per_vertex:
            # Compute per-vertex normals
            mesh.compute_vertex_normals()
            normals = np.asarray(mesh.vertex_normals)
        else:
            # Compute per-face normals
            mesh.compute_triangle_normals()
            normals = np.asarray(mesh.triangle_normals)

        print(f"✅ Computed {'vertex' if per_vertex else 'face'} normals: {len(normals)}")

        return normals

    def calculate_mesh_quality_metrics(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray
    ) -> MeshQualityMetrics:
        """
        Calculate mesh quality metrics.

        Args:
            vertices: Mesh vertices [N, 3]
            triangles: Mesh triangles [T, 3]

        Returns:
            MeshQualityMetrics object
        """
        if not TRIMESH_AVAILABLE:
            print("⚠️  trimesh required for quality metrics.")
            return MeshQualityMetrics(
                triangle_count=len(triangles),
                vertex_count=len(vertices),
                edge_count=0,
                is_watertight=False,
                is_manifold=False,
                mean_triangle_aspect_ratio=0,
                min_triangle_area=0,
                max_triangle_area=0,
                surface_area=0,
                volume=0
            )

        # Create trimesh mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

        # Calculate triangle aspect ratios
        aspect_ratios = []
        for tri in triangles:
            v0, v1, v2 = vertices[tri]
            a = np.linalg.norm(v1 - v0)
            b = np.linalg.norm(v2 - v1)
            c = np.linalg.norm(v0 - v2)
            s = (a + b + c) / 2  # Semi-perimeter
            if s > 0 and a > 0 and b > 0 and c > 0:
                area = np.sqrt(s * (s - a) * (s - b) * (s - c))
                max_edge = max(a, b, c)
                aspect_ratio = max_edge / (2 * area) if area > 0 else 0
                aspect_ratios.append(aspect_ratio)

        # Calculate triangle areas
        triangle_areas = mesh.area_faces

        # Build metrics
        metrics = MeshQualityMetrics(
            triangle_count=len(triangles),
            vertex_count=len(vertices),
            edge_count=len(mesh.edges_unique),
            is_watertight=mesh.is_watertight,
            is_manifold=mesh.is_winding_consistent,
            mean_triangle_aspect_ratio=float(np.mean(aspect_ratios)) if aspect_ratios else 0,
            min_triangle_area=float(np.min(triangle_areas)) if len(triangle_areas) > 0 else 0,
            max_triangle_area=float(np.max(triangle_areas)) if len(triangle_areas) > 0 else 0,
            surface_area=float(mesh.area),
            volume=float(abs(mesh.volume)) if mesh.is_watertight else 0
        )

        print(f"✅ Mesh quality: {metrics.triangle_count} triangles, "
              f"watertight={metrics.is_watertight}, manifold={metrics.is_manifold}")

        return metrics

    def calculate_triangle_statistics(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray
    ) -> Dict[str, int]:
        """
        Calculate triangle count statistics.

        Args:
            vertices: Mesh vertices [N, 3]
            triangles: Mesh triangles [T, 3]

        Returns:
            Statistics dict
        """
        if TRIMESH_AVAILABLE:
            mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
            edge_count = len(mesh.edges_unique)
        else:
            # Estimate edge count (each triangle has 3 edges, shared edges counted once)
            edge_count = len(triangles) * 3 // 2

        stats = {
            'triangle_count': len(triangles),
            'vertex_count': len(vertices),
            'edge_count': edge_count,
        }

        print(f"✅ Triangle statistics: {stats['triangle_count']} triangles, "
              f"{stats['vertex_count']} vertices, {stats['edge_count']} edges")

        return stats

    def validate_mesh_is_watertight(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray
    ) -> bool:
        """
        Validate mesh is watertight.

        All edges should have exactly 2 adjacent faces.

        Args:
            vertices: Mesh vertices [N, 3]
            triangles: Mesh triangles [T, 3]

        Returns:
            True if mesh is watertight
        """
        if not TRIMESH_AVAILABLE:
            print("⚠️  trimesh required for watertight validation.")
            return False

        # Create trimesh mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

        # Check if watertight
        is_watertight = mesh.is_watertight

        print(f"✅ Watertight check: {is_watertight}")

        return is_watertight

    # ========================================================================
    # MESH EXPORT
    # ========================================================================

    def export_to_obj(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        normals: Optional[np.ndarray] = None,
        colors: Optional[np.ndarray] = None,
        output_path: Path = Path("mesh.obj")
    ):
        """
        Export mesh to OBJ format.

        Args:
            vertices: Mesh vertices [N, 3]
            triangles: Mesh triangles [T, 3]
            normals: Optional normals [N, 3]
            colors: Optional colors [N, 3]
            output_path: Output file path
        """
        if OPEN3D_AVAILABLE:
            # Use Open3D for export
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(triangles)

            if normals is not None:
                mesh.vertex_normals = o3d.utility.Vector3dVector(normals)

            if colors is not None:
                mesh.vertex_colors = o3d.utility.Vector3dVector(colors / 255.0)

            o3d.io.write_triangle_mesh(str(output_path), mesh)

        elif TRIMESH_AVAILABLE:
            # Use trimesh for export
            mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
            mesh.export(str(output_path))

        else:
            print("⚠️  Open3D or trimesh required for OBJ export.")
            return

        print(f"✅ Exported mesh to OBJ: {output_path}")

    def export_to_stl(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        output_path: Path = Path("mesh.stl"),
        binary: bool = True
    ):
        """
        Export mesh to STL format (for 3D printing).

        Args:
            vertices: Mesh vertices [N, 3]
            triangles: Mesh triangles [T, 3]
            output_path: Output file path
            binary: If True, export binary STL; else ASCII
        """
        if TRIMESH_AVAILABLE:
            mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
            mesh.export(str(output_path), file_type='stl')
        elif OPEN3D_AVAILABLE:
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(triangles)
            o3d.io.write_triangle_mesh(str(output_path), mesh)
        else:
            print("⚠️  trimesh or Open3D required for STL export.")
            return

        print(f"✅ Exported mesh to STL: {output_path}")

    def export_to_ply_with_faces(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        normals: Optional[np.ndarray] = None,
        colors: Optional[np.ndarray] = None,
        output_path: Path = Path("mesh.ply"),
        binary: bool = True
    ):
        """
        Export mesh to PLY format with faces.

        Args:
            vertices: Mesh vertices [N, 3]
            triangles: Mesh triangles [T, 3]
            normals: Optional normals [N, 3]
            colors: Optional colors [N, 3]
            output_path: Output file path
            binary: If True, export binary; else ASCII
        """
        if not OPEN3D_AVAILABLE:
            print("⚠️  Open3D required for PLY export.")
            return

        # Create Open3D mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)

        if normals is not None:
            mesh.vertex_normals = o3d.utility.Vector3dVector(normals)

        if colors is not None:
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors / 255.0)

        # Export
        o3d.io.write_triangle_mesh(str(output_path), mesh, write_ascii=not binary)

        print(f"✅ Exported mesh to PLY: {output_path}")

    def export_to_collada(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        output_path: Path = Path("mesh.dae")
    ):
        """
        Export mesh to COLLADA format.

        Args:
            vertices: Mesh vertices [N, 3]
            triangles: Mesh triangles [T, 3]
            output_path: Output file path
        """
        if not TRIMESH_AVAILABLE:
            print("⚠️  trimesh required for COLLADA export.")
            return

        # Create trimesh mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

        # Export to COLLADA
        mesh.export(str(output_path), file_type='dae')

        print(f"✅ Exported mesh to COLLADA: {output_path}")

    def export_to_off(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        output_path: Path = Path("mesh.off")
    ):
        """
        Export mesh to OFF (Object File Format).

        Simple text format widely supported by mesh tools.

        Args:
            vertices: Mesh vertices [N, 3]
            triangles: Mesh triangles [T, 3]
            output_path: Output file path
        """
        if TRIMESH_AVAILABLE:
            mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
            mesh.export(str(output_path), file_type='off')
        else:
            # Manual OFF export
            with open(output_path, 'w') as f:
                f.write("OFF\n")
                f.write(f"{len(vertices)} {len(triangles)} 0\n")

                # Write vertices
                for v in vertices:
                    f.write(f"{v[0]} {v[1]} {v[2]}\n")

                # Write faces
                for t in triangles:
                    f.write(f"3 {t[0]} {t[1]} {t[2]}\n")

        print(f"✅ Exported mesh to OFF: {output_path}")
