"""SCIP Symbol ID Format Extraction.

SCIP (Source Code Intelligence Protocol) provides a standard format for
representing code symbols across different languages.

Symbol ID format: <scheme> <package> <descriptor>+

Where:
- scheme: Language-specific scheme (e.g., scip-python, scip-typescript)
- package: Package/module path
- descriptor: One or more descriptors with suffixes:
  - `/` = namespace (module/package)
  - `#` = type (class/interface)
  - `.` = term (function/method/variable)
  - `)` = parameter
  - `]` = type parameter
  - `:` = meta descriptor
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from openmemory.api.indexing.ast_parser import Language, Symbol, SymbolType


# =============================================================================
# Exceptions
# =============================================================================


class InvalidSymbolError(Exception):
    """Raised when a symbol cannot be converted to SCIP format."""

    pass


# =============================================================================
# SCIP Scheme
# =============================================================================


class SCIPScheme(Enum):
    """SCIP scheme for different languages."""

    LOCAL = "local"
    SCIP_PYTHON = "scip-python"
    SCIP_TYPESCRIPT = "scip-typescript"
    SCIP_JAVA = "scip-java"
    SCIP_GO = "scip-go"

    @classmethod
    def for_language(cls, language: Language) -> "SCIPScheme":
        """Get scheme for language."""
        mapping = {
            Language.PYTHON: cls.SCIP_PYTHON,
            Language.TYPESCRIPT: cls.SCIP_TYPESCRIPT,
            Language.TSX: cls.SCIP_TYPESCRIPT,
            Language.JAVA: cls.SCIP_JAVA,
            Language.GO: cls.SCIP_GO,
        }
        return mapping.get(language, cls.LOCAL)


# =============================================================================
# SCIP Descriptor
# =============================================================================


@dataclass(frozen=True)
class SCIPDescriptor:
    """A SCIP descriptor component.

    Descriptors have a name and a suffix indicating the kind:
    - `/` = namespace
    - `#` = type
    - `.` = term (function/method/variable)
    - `)` = parameter
    - `]` = type parameter
    - `:` = meta
    """

    name: str
    suffix: str

    def __str__(self) -> str:
        return f"{self.name}{self.suffix}"

    @classmethod
    def namespace(cls, name: str) -> "SCIPDescriptor":
        """Create namespace descriptor."""
        return cls(name=name, suffix="/")

    @classmethod
    def type_(cls, name: str) -> "SCIPDescriptor":
        """Create type descriptor (class/interface)."""
        return cls(name=name, suffix="#")

    @classmethod
    def term(cls, name: str) -> "SCIPDescriptor":
        """Create term descriptor (function/variable)."""
        return cls(name=name, suffix=".")

    @classmethod
    def method(cls, name: str) -> "SCIPDescriptor":
        """Create method descriptor."""
        return cls(name=name, suffix=".")

    @classmethod
    def parameter(cls, name: str) -> "SCIPDescriptor":
        """Create parameter descriptor."""
        return cls(name=name, suffix=")")

    @classmethod
    def type_parameter(cls, name: str) -> "SCIPDescriptor":
        """Create type parameter descriptor."""
        return cls(name=name, suffix="]")

    @classmethod
    def meta(cls, name: str) -> "SCIPDescriptor":
        """Create meta descriptor."""
        return cls(name=name, suffix=":")


# =============================================================================
# SCIP Symbol ID
# =============================================================================


@dataclass(frozen=True)
class SCIPSymbolID:
    """SCIP symbol identifier.

    Format: <scheme> <package> <descriptor>+
    """

    scheme: SCIPScheme
    package: str
    descriptors: tuple[SCIPDescriptor, ...]

    def __init__(
        self,
        scheme: SCIPScheme,
        package: str,
        descriptors: list[SCIPDescriptor] | tuple[SCIPDescriptor, ...],
    ):
        # Use object.__setattr__ for frozen dataclass
        object.__setattr__(self, "scheme", scheme)
        object.__setattr__(self, "package", package)
        object.__setattr__(
            self,
            "descriptors",
            tuple(descriptors) if isinstance(descriptors, list) else descriptors,
        )

    def __str__(self) -> str:
        descriptor_str = "".join(str(d) for d in self.descriptors)
        return f"{self.scheme.value} {self.package} {descriptor_str}"

    def __hash__(self) -> int:
        return hash((self.scheme, self.package, self.descriptors))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SCIPSymbolID):
            return False
        return (
            self.scheme == other.scheme
            and self.package == other.package
            and self.descriptors == other.descriptors
        )

    @classmethod
    def parse(cls, id_str: str) -> "SCIPSymbolID":
        """Parse symbol ID from string format."""
        parts = id_str.split(" ", 2)
        if len(parts) < 3:
            raise InvalidSymbolError(f"Invalid symbol ID format: {id_str}")

        scheme_str, package, descriptor_str = parts

        # Parse scheme
        try:
            scheme = SCIPScheme(scheme_str)
        except ValueError:
            raise InvalidSymbolError(f"Unknown scheme: {scheme_str}")

        # Parse descriptors
        descriptors = cls._parse_descriptors(descriptor_str)

        return cls(scheme=scheme, package=package, descriptors=descriptors)

    @staticmethod
    def _parse_descriptors(descriptor_str: str) -> list[SCIPDescriptor]:
        """Parse descriptor string into list of descriptors."""
        descriptors = []
        current_name = ""

        for char in descriptor_str:
            if char in "/#.)]":
                if current_name:
                    descriptors.append(SCIPDescriptor(name=current_name, suffix=char))
                    current_name = ""
            elif char == ":":
                if current_name:
                    descriptors.append(SCIPDescriptor(name=current_name, suffix=":"))
                    current_name = ""
            else:
                current_name += char

        return descriptors


# =============================================================================
# Symbol ID Builder
# =============================================================================


class SymbolIDBuilder:
    """Fluent builder for SCIP symbol IDs."""

    def __init__(self):
        self._scheme: SCIPScheme = SCIPScheme.LOCAL
        self._package: str = ""
        self._descriptors: list[SCIPDescriptor] = []

    def scheme(self, scheme: SCIPScheme) -> "SymbolIDBuilder":
        """Set scheme."""
        self._scheme = scheme
        return self

    def package(self, package: str) -> "SymbolIDBuilder":
        """Set package."""
        self._package = package
        return self

    def namespace(self, name: str) -> "SymbolIDBuilder":
        """Add namespace descriptor."""
        self._descriptors.append(SCIPDescriptor.namespace(name))
        return self

    def type_(self, name: str) -> "SymbolIDBuilder":
        """Add type descriptor."""
        self._descriptors.append(SCIPDescriptor.type_(name))
        return self

    def term(self, name: str) -> "SymbolIDBuilder":
        """Add term descriptor."""
        self._descriptors.append(SCIPDescriptor.term(name))
        return self

    def method(self, name: str) -> "SymbolIDBuilder":
        """Add method descriptor."""
        self._descriptors.append(SCIPDescriptor.method(name))
        return self

    def parameter(self, name: str) -> "SymbolIDBuilder":
        """Add parameter descriptor."""
        self._descriptors.append(SCIPDescriptor.parameter(name))
        return self

    def type_parameter(self, name: str) -> "SymbolIDBuilder":
        """Add type parameter descriptor."""
        self._descriptors.append(SCIPDescriptor.type_parameter(name))
        return self

    def meta(self, name: str) -> "SymbolIDBuilder":
        """Add meta descriptor."""
        self._descriptors.append(SCIPDescriptor.meta(name))
        return self

    def build(self) -> SCIPSymbolID:
        """Build symbol ID."""
        return SCIPSymbolID(
            scheme=self._scheme,
            package=self._package,
            descriptors=self._descriptors,
        )


# =============================================================================
# Package Resolvers
# =============================================================================


class PackageResolver:
    """Base class for package resolution."""

    def resolve(self, file_path: Path, root_path: Path) -> str:
        """Resolve package name from file path."""
        raise NotImplementedError


class PythonPackageResolver(PackageResolver):
    """Python package resolver."""

    def resolve(self, file_path: Path, root_path: Path) -> str:
        """Resolve Python package from file path."""
        try:
            rel_path = file_path.relative_to(root_path)
        except ValueError:
            return file_path.stem

        # Convert path to module name
        parts = list(rel_path.parts)

        # Handle __init__.py
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            # Remove .py extension
            parts[-1] = parts[-1].replace(".py", "")

        return ".".join(parts)


class TypeScriptPackageResolver(PackageResolver):
    """TypeScript package resolver."""

    def resolve(self, file_path: Path, root_path: Path) -> str:
        """Resolve TypeScript package from file path."""
        # Try to read package.json for package name
        package_json = root_path / "package.json"
        package_name = ""

        if package_json.exists():
            try:
                data = json.loads(package_json.read_text())
                package_name = data.get("name", "")
            except (json.JSONDecodeError, IOError):
                pass

        try:
            rel_path = file_path.relative_to(root_path)
        except ValueError:
            return file_path.stem

        # Convert path to module path (without extension)
        path_str = str(rel_path).replace(".ts", "").replace(".tsx", "")

        if package_name:
            return f"{package_name}/{path_str}"
        return path_str


class JavaPackageResolver(PackageResolver):
    """Java package resolver."""

    def resolve(self, file_path: Path, root_path: Path) -> str:
        """Resolve Java package from source file."""
        # Read the file and find package declaration
        try:
            content = file_path.read_text()
            match = re.search(r"^package\s+([\w.]+);", content, re.MULTILINE)
            if match:
                return match.group(1)
        except IOError:
            pass

        return ""  # Default package


class GoPackageResolver(PackageResolver):
    """Go package resolver."""

    def __init__(self):
        self._module_path: Optional[str] = None

    def _read_module_path(self, root_path: Path) -> Optional[str]:
        go_mod = root_path / "go.mod"
        if not go_mod.exists():
            return None
        try:
            for line in go_mod.read_text().splitlines():
                line = line.strip()
                if line.startswith("module "):
                    return line.split(" ", 1)[1].strip()
        except IOError:
            return None
        return None

    def resolve(self, file_path: Path, root_path: Path) -> str:
        """Resolve Go package from source file."""
        if self._module_path is None:
            self._module_path = self._read_module_path(root_path)

        try:
            rel_path = file_path.relative_to(root_path)
        except ValueError:
            rel_path = file_path

        rel_dir = rel_path.parent
        if str(rel_dir) in (".", ""):
            return self._module_path or ""
        if self._module_path:
            return f"{self._module_path}/{rel_dir.as_posix()}"
        return rel_dir.as_posix()


# =============================================================================
# SCIP Symbol Extractor
# =============================================================================


class SCIPSymbolExtractor:
    """Extracts SCIP symbol IDs from parsed symbols."""

    def __init__(self, root_path: Path):
        self.root_path = root_path
        self._resolvers = {
            Language.PYTHON: PythonPackageResolver(),
            Language.TYPESCRIPT: TypeScriptPackageResolver(),
            Language.TSX: TypeScriptPackageResolver(),
            Language.JAVA: JavaPackageResolver(),
            Language.GO: GoPackageResolver(),
        }

    def extract(self, symbol: Symbol, file_path: Path) -> SCIPSymbolID:
        """Extract SCIP symbol ID from a symbol."""
        scheme = SCIPScheme.for_language(symbol.language)
        resolver = self._resolvers.get(symbol.language)

        if resolver:
            package = resolver.resolve(file_path, self.root_path)
        else:
            package = ""

        # Build descriptors
        builder = SymbolIDBuilder().scheme(scheme).package(package)

        # Add namespace from file path
        namespace = self._get_namespace(file_path)
        if namespace:
            builder.namespace(namespace)

        # Add parent type if this is a method or nested class
        if symbol.parent_name:
            builder.type_(symbol.parent_name)

        # Add the symbol descriptor based on type
        if symbol.symbol_type == SymbolType.CLASS:
            builder.type_(symbol.name)
        elif symbol.symbol_type == SymbolType.INTERFACE:
            builder.type_(symbol.name)
        elif symbol.symbol_type == SymbolType.ENUM:
            builder.type_(symbol.name)
        elif symbol.symbol_type == SymbolType.METHOD:
            builder.method(symbol.name)
        elif symbol.symbol_type == SymbolType.FUNCTION:
            builder.term(symbol.name)
        elif symbol.symbol_type == SymbolType.FIELD:
            builder.meta("field").term(symbol.name)
        elif symbol.symbol_type == SymbolType.PROPERTY:
            builder.meta("property").term(symbol.name)
        elif symbol.symbol_type == SymbolType.VARIABLE:
            builder.term(symbol.name)
        elif symbol.symbol_type == SymbolType.IMPORT:
            builder.term(symbol.name)
        elif symbol.symbol_type == SymbolType.TYPE_ALIAS:
            builder.type_(symbol.name)
        else:
            builder.term(symbol.name)

        return builder.build()

    def _get_namespace(self, file_path: Path) -> str:
        """Get namespace from file path."""
        try:
            rel_path = file_path.relative_to(self.root_path)
        except ValueError:
            return file_path.stem

        # Remove extension and use stem
        return rel_path.stem


# =============================================================================
# Factory Function
# =============================================================================


def create_extractor(root_path: Path) -> SCIPSymbolExtractor:
    """Create a SCIP symbol extractor."""
    return SCIPSymbolExtractor(root_path)
