import io
        ascending=False,
    ),
    use_container_width=True,
)

# -------------------------------------------------
# Filtered compounds
# -------------------------------------------------
st.subheader("✅ Passing Compounds")

st.dataframe(
    pass_df.sort_values(
        by="CNS_MPO_like",
        ascending=False,
    ),
    use_container_width=True,
)

# -------------------------------------------------
# Excel export
# -------------------------------------------------
excel_buffer = io.BytesIO()

with pd.ExcelWriter(
    excel_buffer,
    engine="openpyxl",
) as writer:

    df_out.to_excel(
        writer,
        index=False,
        sheet_name="All_Compounds",
    )

    pass_df.to_excel(
        writer,
        index=False,
        sheet_name="Passing_Compounds",
    )

excel_buffer.seek(0)

# -------------------------------------------------
# Download button
# -------------------------------------------------
st.download_button(
    label="⬇ Download Excel (.xlsx)",
    data=excel_buffer,
    file_name="cns_mpo_results.xlsx",
    mime=(
        "application/vnd.openxmlformats-officedocument."
        "spreadsheetml.sheet"
    ),
)

# -------------------------------------------------
# Plot
# -------------------------------------------------
if len(valid_df) > 0:

    st.subheader("📊 Score Distribution")

    hist_df = (
        valid_df["CNS_MPO_like"]
        .round(1)
        .value_counts()
        .sort_index()
    )

    st.bar_chart(hist_df)