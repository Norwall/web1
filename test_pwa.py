import flet as ft

def main(page: ft.Page):

    main_page = ft.Column(
        [
            ft.Row(
                [
                    ft.TextField(),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
            ),
        ],
        alignment=ft.MainAxisAlignment.CENTER,
    )

    page.add(main_page)


ft.app(target=main, view=ft.WEB_BROWSER)
