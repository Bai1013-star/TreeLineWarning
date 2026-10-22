"""Read-only extent checks; never repair a LAS by silently dropping points."""
from pathlib import Path


class PointCloudIntegrityError(ValueError):
    """A readable LAS header declares point bytes absent from the local file."""


def validate_las_extent(source: Path) -> dict | None:
    import laspy

    source = Path(source)
    # Header-only reading must not trigger a full cloud read or EVLR seeking.
    try:
        with source.open('rb') as stream:
            header = laspy.LasHeader.read_from(stream, read_evlrs=False)
    except (OSError, ValueError, EOFError, laspy.errors.LaspyException):
        # Preserve the existing capability inspector's handling of unreadable
        # headers; this check specifically detects a provably truncated extent.
        return None
    if header.are_points_compressed:
        return None  # LAZ compressed bytes do not equal count * record size.
    size = source.stat().st_size
    offset, count, record = int(header.offset_to_point_data), int(header.point_count), int(header.point_format.size)
    expected = offset + count * record
    point_end = size
    if header.number_of_evlrs and header.start_of_first_evlr:
        point_end = min(point_end, int(header.start_of_first_evlr))
    if expected > point_end:
        complete = max(0, (point_end - offset) // record)
        raise PointCloudIntegrityError(
            f'原始 LAS 文件不完整：{source.name}。文件头声明 {count:,} 个点，'
            f'当前文件最多容纳 {complete:,} 条完整点记录；点数据至少缺少 {expected-point_end:,} 字节。'
            '这不是断网或梯子问题。请从原电脑重新复制或重新导出完整 LAS/LAZ，'
            '再点“导入 LAS/LAZ”替换；项目保留，原文件未修改，不能补造缺失点。'
        )
    return dict(declared_point_count=count, point_record_bytes=record,
                point_data_offset=offset, expected_point_data_end=expected,
                file_size_bytes=size, status='COMPLETE_EXTENT')
