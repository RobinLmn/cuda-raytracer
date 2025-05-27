#include "file_utility.hpp"

#include <windows.h>
#include <commdlg.h>

namespace core
{
	std::string new_file_dialog(const char* initial_filename, const char* initial_directory, const char* files_filter)
	{
		char filename[MAX_PATH] = {0};
		strcpy_s(filename, initial_filename);

		OPENFILENAMEA ofn{};
		ofn.lStructSize = sizeof(ofn);
		ofn.hwndOwner = GetActiveWindow();
		ofn.lpstrFilter = files_filter;
		ofn.lpstrFile = filename;
		ofn.nMaxFile = sizeof(filename);
		ofn.lpstrInitialDir = initial_directory;
		ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

		if (GetSaveFileNameA(&ofn))
		{
			return filename;
		}

		return {};
	}
}
