import { useEffect, useRef } from 'react'
import 'datatables.net-dt/css/dataTables.dataTables.css'
import DataTable from 'datatables.net-dt'

interface Column {
  title: string
  data: string
  render?: (data: any, type: string, row: any) => string
  className?: string
}

interface DataTableProps {
  columns: Column[]
  data: any[]
  defaultPageLength?: number
  pageLengthOptions?: number[]
  className?: string
}

export function CustomDataTable({
  columns,
  data,
  defaultPageLength = 50,
  pageLengthOptions = [50, 100, 1000],
  className = '',
}: DataTableProps) {
  const tableRef = useRef<HTMLTableElement>(null)
  const dataTableRef = useRef<any>(null)

  useEffect(() => {
    if (!tableRef.current) return

    // Initialize DataTable
    if (!dataTableRef.current) {
      dataTableRef.current = new DataTable(tableRef.current, {
        columns,
        data,
        pageLength: defaultPageLength,
        lengthMenu: pageLengthOptions,
        order: [[0, 'asc']], // Default sort by first column
        responsive: true,
        autoWidth: false,
        language: {
          search: 'Search:',
          lengthMenu: 'Show _MENU_ entries per page',
          info: 'Showing _START_ to _END_ of _TOTAL_ entries',
          infoEmpty: 'No entries available',
          infoFiltered: '(filtered from _TOTAL_ total entries)',
          zeroRecords: 'No matching records found',
          emptyTable: 'No data available in table',
        },
        // Stripe table rows (alternating colors)
        stripeClasses: ['bg-white', 'bg-gray-50'],
      })
    } else {
      // Update data if table already exists
      dataTableRef.current.clear()
      dataTableRef.current.rows.add(data)
      dataTableRef.current.draw()
    }

    // Cleanup on unmount
    return () => {
      if (dataTableRef.current) {
        dataTableRef.current.destroy()
        dataTableRef.current = null
      }
    }
  }, [data, columns, defaultPageLength, pageLengthOptions])

  return (
    <div className={`datatable-wrapper ${className}`}>
      <table ref={tableRef} className="display stripe hover" style={{ width: '100%' }}>
        <thead>
          <tr>
            {columns.map((col, idx) => (
              <th key={idx}>{col.title}</th>
            ))}
          </tr>
        </thead>
        <tbody></tbody>
      </table>

      <style>{`
        /* DataTables custom styling */
        .datatable-wrapper {
          font-size: 0.875rem;
        }

        .datatable-wrapper table.dataTable {
          border-collapse: collapse;
          width: 100%;
        }

        .datatable-wrapper table.dataTable thead th {
          background-color: #f9fafb;
          border-bottom: 2px solid #e5e7eb;
          padding: 0.75rem 1rem;
          text-align: left;
          font-weight: 600;
          color: #374151;
        }

        .datatable-wrapper table.dataTable tbody tr {
          border-bottom: 1px solid #e5e7eb;
        }

        .datatable-wrapper table.dataTable tbody tr.odd {
          background-color: #ffffff;
        }

        .datatable-wrapper table.dataTable tbody tr.even {
          background-color: #f9fafb;
        }

        .datatable-wrapper table.dataTable tbody tr:hover {
          background-color: #f3f4f6;
        }

        .datatable-wrapper table.dataTable tbody td {
          padding: 0.75rem 1rem;
          color: #1f2937;
        }

        /* Pagination styling */
        .datatable-wrapper .dataTables_wrapper .dataTables_paginate {
          padding-top: 1rem;
        }

        .datatable-wrapper .dataTables_wrapper .dataTables_paginate .paginate_button {
          padding: 0.25rem 0.75rem;
          margin: 0 0.125rem;
          border: 1px solid #d1d5db;
          border-radius: 0.25rem;
          background-color: #ffffff;
          color: #374151;
          cursor: pointer;
        }

        .datatable-wrapper .dataTables_wrapper .dataTables_paginate .paginate_button:hover {
          background-color: #f3f4f6;
        }

        .datatable-wrapper .dataTables_wrapper .dataTables_paginate .paginate_button.current {
          background-color: #3b82f6;
          color: #ffffff;
          border-color: #3b82f6;
        }

        .datatable-wrapper .dataTables_wrapper .dataTables_paginate .paginate_button.disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        /* Search box styling */
        .datatable-wrapper .dataTables_wrapper .dataTables_filter input {
          padding: 0.5rem;
          border: 1px solid #d1d5db;
          border-radius: 0.375rem;
          margin-left: 0.5rem;
        }

        /* Length selector styling */
        .datatable-wrapper .dataTables_wrapper .dataTables_length select {
          padding: 0.375rem 2rem 0.375rem 0.75rem;
          border: 1px solid #d1d5db;
          border-radius: 0.375rem;
          margin: 0 0.5rem;
        }

        /* Info text */
        .datatable-wrapper .dataTables_wrapper .dataTables_info {
          padding-top: 1rem;
          color: #6b7280;
          font-size: 0.875rem;
        }

        /* Text alignment classes */
        .datatable-wrapper table.dataTable td.dt-right {
          text-align: right;
        }

        .datatable-wrapper table.dataTable td.dt-center {
          text-align: center;
        }
      `}</style>
    </div>
  )
}
