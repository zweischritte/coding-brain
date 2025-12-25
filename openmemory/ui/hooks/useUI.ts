import { useCallback } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { AppDispatch, RootState } from '@/store/store';
import { openUpdateMemoryDialog, closeUpdateMemoryDialog } from '@/store/uiSlice';

export const useUI = () => {
  const dispatch = useDispatch<AppDispatch>();
  const updateMemoryDialog = useSelector((state: RootState) => state.ui.dialogs.updateMemory);

  const handleOpenUpdateMemoryDialog = useCallback((memoryId: string, memoryContent: string) => {
    dispatch(openUpdateMemoryDialog({ memoryId, memoryContent }));
  }, [dispatch]);

  const handleCloseUpdateMemoryDialog = useCallback((open?: boolean) => {
    // Only close if explicitly closed (open === false) or called without argument
    if (open === false || open === undefined) {
      dispatch(closeUpdateMemoryDialog());
    }
  }, [dispatch]);

  return {
    updateMemoryDialog,
    handleOpenUpdateMemoryDialog,
    handleCloseUpdateMemoryDialog,
  };
}; 